#!/usr/bin/env bash

strix_gate_ticket_matches() {
    local ticket=$1 expected=$2 actual
    [[ -s $ticket ]] || return 1
    IFS= read -r actual < "$ticket" || return 1
    [[ $actual == "$expected" ]]
}
