// SPDX-License-Identifier: Apache-2.0
// Host-only regression for scripts/patches/kvaf-reuse-device-ra.patch.
/* Run after applying the KVAF patch:
   c++ -std=c++17 -Ithird_party/KVAF/access_operator/include \
       -Ithird_party/KVAF/access_operator/src tests/ut/kvaf/test_native_session.cc \
       -ldl -pthread -o /tmp/test-kvaf-native-session
   /tmp/test-kvaf-native-session
*/
#include <cassert>
#include <memory>

#include "direct_hccp/direct_hccp_session.cc"

int main() {
  using namespace kvaf::direct_hccp;
  DirectHccpSession borrowed;
  borrowed.key = {3U, 18U};
  borrowed.ra_init = [](RaInitConfig* config) {
    // This models HCCL's live device RA: a second offline init is forbidden.
    assert(config->nic_position == NetworkMode::kPeerOnline);
    return 0;
  };
  borrowed.ra_deinit = [](RaInitConfig* config) {
    assert(config->nic_position == NetworkMode::kPeerOnline);
    return 0;
  };
  borrowed.rt_open_net_service = [](RtNetServiceOpenArgs*) { return 0; };
  assert(Bootstrap(&borrowed, true));
  assert(borrowed.peer_ra_owned && !borrowed.device_ra_owned);
  DestroySession(&borrowed);  // Must not deinitialize HCCL's device RA.

  DirectHccpSession standalone;
  standalone.ra_init = [](RaInitConfig*) { return 0; };
  standalone.ra_deinit = [](RaInitConfig*) { return 0; };
  standalone.rt_open_net_service = [](RtNetServiceOpenArgs*) { return 0; };
  assert(Bootstrap(&standalone, false));
  assert(standalone.device_ra_owned);
  DestroySession(&standalone);

  DirectHccpSession failed;
  failed.ra_init = [](RaInitConfig* config) {
    assert(config->nic_position == NetworkMode::kPeerOnline);
    return 0;
  };
  failed.ra_deinit = [](RaInitConfig* config) {
    assert(config->nic_position == NetworkMode::kPeerOnline);
    return 0;
  };
  failed.rt_open_net_service = [](RtNetServiceOpenArgs*) { return 1; };
  assert(!Bootstrap(&failed, true));  // No fallback to device RaInit on error.
  assert(!failed.peer_ra_owned && !failed.device_ra_owned);
  DestroySession(&failed);

  // Priming releases its lease at refs=0. Get and Put must share the retained
  // session without restarting RA, even when they use the default argument.
  auto cached = std::make_unique<DirectHccpSession>();
  cached->key = {3U, 18U};
  auto* expected = cached.get();
  const SessionKey key = cached->key;
  g_sessions.emplace(key, std::move(cached));
  DirectHccpSession *get = nullptr, *put = nullptr;
  assert(AcquireDirectHccpSession(3U, 18U, &get));
  assert(AcquireDirectHccpSession(3U, 18U, &put));
  assert(get == expected && put == expected && expected->reference_count == 2U);
  ReleaseDirectHccpSession(get);
  ReleaseDirectHccpSession(put);
  assert(expected->reference_count == 0U);
  assert(g_sessions.at({3U, 18U}).get() == expected);
  g_sessions.clear();
}
