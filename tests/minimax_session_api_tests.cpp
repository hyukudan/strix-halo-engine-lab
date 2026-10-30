#include "minimax/serve/minimax_session.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <stdexcept>
using namespace strix::serve;
void check(bool value) { if (!value) throw std::runtime_error("session API assertion failed"); }
int main() {
    try {
        check(session_command("GET", "/v1/sessions", "").find("op")->as_string() == "list");
        check(session_command("GET", "/v1/cache/stats", "").find("op")->as_string() == "stats");
        for (const char* op : {"park", "resume"}) {
            const auto cmd = session_command("POST", std::string("/v1/sessions/Ab_09-/") + op, "{}");
            check(cmd.find("op")->as_string() == op && cmd.find("id")->as_string() == "Ab_09-");
        }
        check(session_command("DELETE", "/v1/sessions/a", "").find("op")->as_string() == "delete");
        check(session_command("POST", "/v1/sessions/a/checkpoint", R"({"durability":"ram"})").find("op")->as_string() == "ram_checkpoint");
        check(session_command("POST", "/v1/sessions/a/checkpoint", R"({"durability":"disk"})").find("op")->as_string() == "checkpoint");
        auto refuses = [](const char* method, const std::string& path, const std::string& body, int status) {
            try { (void)session_command(method,path,body); } catch (const SessionApiError& e) { check(e.status == status); return; }
            throw std::runtime_error("invalid route accepted");
        };
        refuses("POST","/v1/session-cache",R"({"operation":"list"})",404);
        refuses("POST","/v1/sessions/a/checkpoint",R"({"durability":"unknown"})",400);
        refuses("POST","/v1/sessions/a/checkpoint","{",400);
        for (const auto& id : {std::string(), std::string("bad.id"), std::string("%2e"), std::string(129,'x')})
            refuses("POST","/v1/sessions/"+id+"/park","",400);
        for (const auto& pair : {std::pair{"session_not_found",404}, {"session_budget",507}, {"session_incompatible",409}, {"session_corrupt",422}}) {
            const auto error = session_failure(pair.first); check(error.status == pair.second && error.code == pair.first);
            check(session_error(error.code,error.what()).find("error")->find("message")->is_string());
        }
        const auto peer_refusal = session_error("session_budget", "peer session precondition failed", 507);
        check(peer_refusal.find("_status")->as_int() == 507);
        check(peer_refusal.find("error")->find("code")->as_string() == "session_budget");
        // The HTTP writer uses _status and serializes only the public error.
        check(!session_error("session_budget", "peer session precondition failed").find("_status"));
        namespace ms = strix::minimax::serve;
        ms::SessionTasks tasks;
        auto list = Json::object(), stats = Json::object();
        list.set("sessions", Json::array()); stats.set("parked", Json(std::uint64_t(7)));
        tasks.publish_metadata(list, stats);
        const auto start = std::chrono::steady_clock::now();
        try {
            tasks.submit(session_command("POST", "/v1/sessions/a/park", ""), std::chrono::milliseconds(20));
            check(false);
        } catch (const SessionApiError& e) {
            check(e.status == 503 && e.code == "unavailable");
            check(std::string(e.what()).find("retry after 1 second") != std::string::npos);
        }
        check(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
        check(!tasks.pending && !tasks.take()); // expired queued tasks cannot run later
        // Reads stay available while a mutation is queued, without any engine boundary.
        std::thread waiter([&] {
            const auto reply = tasks.submit(session_command("DELETE", "/v1/sessions/a", ""));
            check(reply.find("done")->as_bool());
        });
        while (!tasks.pending) std::this_thread::yield();
        check(tasks.submit(session_command("GET", "/v1/sessions", "")).find("sessions")->is_array());
        check(tasks.submit(session_command("GET", "/v1/cache/stats", "")).find("parked")->as_int() == 7);
        auto task = tasks.take(); auto done = Json::object(); done.set("done", Json(true));
        task->result.set_value(done); waiter.join();
        // Queue expiry cannot report busy for an operation the engine has taken.
        auto executing = std::async(std::launch::async, [&] {
            return tasks.submit(session_command("DELETE", "/v1/sessions/a", ""), std::chrono::milliseconds(100));
        });
        while (!tasks.pending) std::this_thread::yield();
        task = tasks.take(); check(task->dispatched);
        check(executing.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout);
        check(!tasks.pending && !tasks.take());
        task->result.set_value(done);
        check(executing.get().find("done")->as_bool());
        std::cout << "MiniMax common session API PASS\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
