The host wire headers in `src/glm/import`, the dimension/state layout headers,
and the CPU receiver/replay fixtures in `tests/glm_native` are adopted from
this project's node import branch at revision
`298ede2f5241d870fc50daa5c7b80a54a052c129`. They retain the project's license.
The production sender uses those wire types directly; it does not implement
a different protocol or change the receiver's accepted identities.
