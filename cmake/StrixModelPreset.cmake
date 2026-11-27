if(NOT TARGET strix_model_preset)
    add_library(strix_model_preset STATIC "${CMAKE_CURRENT_LIST_DIR}/../src/profile/model_preset.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/profile/multi_device.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../src/profile/all_on.cpp")
    # The --all-on lever data: docs/optimizations.json rows exported by tools/ci/export_all_on.py.
    set(_strix_all_on_data "${CMAKE_CURRENT_LIST_DIR}/../data/profile/all_on.json")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_strix_all_on_data}")
    file(READ "${_strix_all_on_data}" STRIX_ALL_ON_JSON)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/AllOnData.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/strix/profile/all_on_data.hpp" @ONLY NEWLINE_STYLE LF)
    unset(STRIX_ALL_ON_JSON)
    unset(_strix_all_on_data)
    target_include_directories(strix_model_preset PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(strix_model_preset PUBLIC strix_model_description strix_serve_json)
    strix_target_warnings(strix_model_preset)
    if(MSVC)
        target_compile_definitions(strix_model_preset PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
