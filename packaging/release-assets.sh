#!/usr/bin/env bash
# Generate the checksum manifest and release notes in the downloaded package directory.
set -euo pipefail
export LC_ALL=C
shopt -s nullglob
cd "${1:?usage: release-assets.sh PACKAGE_DIR}"

packages=()
for component in core tools docs hip; do
    debs=(strix-"$component"_*.deb)
    rpms=(strix-"$component"-*.rpm)
    archives=(strix-"$component"-*.tar.gz)
    if (( ${#debs[@]} != 1 || ${#rpms[@]} != 1 || ${#archives[@]} != 1 )); then
        echo "Expected one DEB, RPM and tar.gz for strix-$component" >&2
        exit 1
    fi
    packages+=("${debs[@]}" "${rpms[@]}" "${archives[@]}")
done
assets=(strix-*.deb strix-*.rpm strix-*.tar.gz)
if (( ${#assets[@]} != ${#packages[@]} )); then
    echo 'Unexpected package in release assets' >&2
    exit 1
fi

hip_archive=(strix-hip-*.tar.gz)
archs=$(tar -xOzf "${hip_archive[0]}" --wildcards '*/share/strix/support/hip-archs.txt')
qualified=$(printf '%s\n' "$archs" | awk '/: qualified$/ { print "- HIP " $0 }')
{
    echo '## Packages'
    printf -- '- `%s`\n' "${packages[@]}"
    if [[ -n "$qualified" ]]; then
        printf '\n## Qualified targets\n%s\n' "$qualified"
    fi
} >release-notes.md
sha256sum --binary -- "${packages[@]}" >SHA256SUMS
