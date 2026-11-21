#!/usr/bin/env bash
# Run as root in a fresh container, with the host package directory mounted read-only.
set -euo pipefail
mode=${1:?usage: clean-host-smoke.sh deb|tgz PACKAGE_DIR}
cd "${2:?expected package directory}"
export DEBIAN_FRONTEND=noninteractive
apt-get update

case "$mode" in
    deb)
        apt-get install -y --no-install-recommends file
        debs=(./strix-core_*.deb ./strix-tools_*.deb ./strix-docs_*.deb)
        # The host components do not own service conffiles. Simulate a site-provided one.
        mkdir -p /etc/strix
        printf '# Local configuration\n' >/tmp/strix-local-config
        cp /tmp/strix-local-config /etc/strix/serve.toml
        apt-get install -y --no-install-recommends "${debs[@]}"
        bash /usr/share/strix/support/smoke-core.sh --require-health
        apt-get remove -y strix-core strix-tools strix-docs
        test ! -e /usr/bin/strix-serve
        cmp /tmp/strix-local-config /etc/strix/serve.toml
        apt-get install -y --no-install-recommends "${debs[@]}"
        cmp /tmp/strix-local-config /etc/strix/serve.toml
        bash /usr/share/strix/support/smoke-core.sh --require-health
        root=/
        ;;
    tgz)
        apt-get install -y --no-install-recommends file libstdc++6 python3-numpy
        root=$(mktemp -d)
        for component in core tools docs; do
            archives=(strix-"$component"-*.tar.gz)
            tar -xzf "${archives[0]}" -C "$root"
        done
        bash "$root/usr/share/strix/support/smoke-core.sh" --root "$root" --require-health
        ;;
    *) echo "Unknown smoke test mode: $mode" >&2; exit 2 ;;
esac

for component in core tools docs; do
    grep -E '^abi-floor: Ubuntu 24\.04 \(glibc 2\.39, libstdc\+\+6 [^)]+\)$' \
        "${root%/}/usr/share/strix/support/build-$component.txt"
done
