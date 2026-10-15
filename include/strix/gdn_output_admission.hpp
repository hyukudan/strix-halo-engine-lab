#pragma once
#include "strix/model_contract.hpp"
#include "strix/model_manifest.hpp"
namespace strix {
// Validate every executed GDN output tensor before decoder setup or capture.
void validate_virtual_q8_gdn_output_manifest(const ModelContract&,const ModelManifest&);
}
