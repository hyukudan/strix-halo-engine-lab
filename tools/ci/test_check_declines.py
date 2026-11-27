"""Fixture tests of tools/ci/check_declines.py."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_declines as cd  # noqa: E402


UNRECORDED = 'void f() {\n    std::cerr << "lever x declined (unmeasured regime); using incumbent weights\\n";\n}\n'
RECORDED = ('void f() {\n    std::cerr << "lever x declined (outside band)\\n";\n'
            '    profile::record_decline({"x", "1", "0", profile::DeclineOrigin::load, "", why}, std::cerr);\n}\n')
COUNTER = 'void f() { std::fprintf(stderr, "draft batch: declined=%llu steps=%llu\\n", a, b); }\n'
NOT_STREAM = 'constexpr const char* kWhy = "declined: not byte-exact";\n'
ENGAGED = ('bool g() {\n    return profile::all_on::engaged("STRIX_X", on, "declined: no pack", std::cerr);\n}\n'
           'void h() { std::cerr << "STRIX_X declined: no pack\\n"; }\n')
MULTILINE = 'void f() {\n    std::fprintf(stderr,\n        "glm seqpar auto declined: %s; legacy path\\n", why);\n}\n'


def check(name, ok):
    if not ok:
        raise AssertionError(name)
    print("ok", name)


def main():
    found, recorded = cd.scan_text("src/a.hpp", UNRECORDED)
    check("an error-stream decline without a record is a site",
          found == ["src/a.hpp|lever x declined (unmeasured regime); using incumbent weights\\n"] and recorded == 0)
    found, recorded = cd.scan_text("src/a.hpp", RECORDED)
    check("a decline next to its record passes", found == [] and recorded == 1)
    check("a counter is not a decline", cd.scan_text("src/a.hpp", COUNTER) == ([], 0))
    check("a literal never written to the error stream is not a site", cd.scan_text("src/a.hpp", NOT_STREAM) == ([], 0))
    found, recorded = cd.scan_text("src/a.hpp", ENGAGED)
    check("the --all-on record counts as recorded", found == [] and recorded == 2)
    found, _ = cd.scan_text("src/a.hip", MULTILINE)
    check("a stream statement opened on an earlier line is found", len(found) == 1)

    key = "src/a.hpp|lever x declined"
    check("a listed site passes", cd.compare([key], [key]) == [])
    check("a new unrecorded site fails", any("unrecorded decline" in f for f in cd.compare([key], [])))
    check("a stale baseline line fails", any("gone or now recorded" in f for f in cd.compare([], [key])))
    check("a duplicated baseline line fails", any("twice" in f for f in cd.compare([key], [key, key])))
    check("the baseline may not grow against its base",
          any("shrink-only" in f for f in cd.compare([key], [key], base_baseline=[])))
    check("the baseline may shrink against its base", cd.compare([key], [key], base_baseline=[key, "src/b|y"]) == [])

    rendered = cd.render_baseline([key, key])
    check("the rendered baseline round-trips one line per site", cd.parse_baseline(rendered) == [key])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src/a.hpp").write_text(UNRECORDED, encoding="utf-8")
        (root / "include/strix/profile").mkdir(parents=True)
        (root / "include/strix/profile/decline.hpp").write_text(UNRECORDED, encoding="utf-8")
        found, _ = cd.scan_tree(root)
        check("the tree scan finds the site and skips the record's own header", [k.split("|")[0] for k in found] == ["src/a.hpp"])
    print("check_declines tests: PASS")


if __name__ == "__main__":
    main()
