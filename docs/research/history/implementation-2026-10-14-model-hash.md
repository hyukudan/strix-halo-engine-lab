# Optional full-file SHA-256 provider

`-DSTRIX_USE_OPENSSL_SHA256=ON` links `OpenSSL::Crypto` and uses its EVP digest
API for the bulk file hash. The option defaults to OFF, preserving the portable
build without that dependency. Provider setup, update and finalization failures
abort verification; they never substitute a cached identity or skip data.

Every file byte is still read. The existing before/after size checks, read-error
handling and canonical model identity framing remain unchanged. The byte hash
oracle and aggregate framing keep the portable SHA-256 implementation, so the
optional provider cannot silently change the model identity format.

Tests cover empty, short and million-byte known vectors, SHA-256 padding/block
boundaries, the one-MiB I/O boundary and multiple I/O chunks. File digests match
the portable byte oracle with the provider enabled and disabled. Both variants
compiled with strict warnings and `-Werror` and passed on Linux. Actual startup
cost requires a separate full-model measurement; this option does not improve
per-token kernel performance by itself.

Dependency/API reference: [OpenSSL EVP digest documentation](https://docs.openssl.org/3.5/man3/EVP_DigestInit/).
