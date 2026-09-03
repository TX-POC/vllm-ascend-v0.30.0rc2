# Stage-B transport dependency

This directory does not vendor a second Direct-HCCP implementation.  Its only
transport dependency is the KVAF source-controlled header:

```text
access_operator/demo/aiv_batch_get/kernel/kvaf_forward_resident_get_device.h
```

The enclosing P10 package must pass that header as the preprocessor macro
`P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER`, and must use include roots
from the exact KVAF revision recorded in the product provenance.  The header
exports the only permitted operations: `BeginKvafForwardResidentBatchGet`,
`TryRetireKvafForwardResidentBatchGet`, and
`FinishKvafForwardResidentBatchGet`.

No source from P8/P9 is imported here.  In particular, there is no host
`execute_batch_get` call, no per-core Get partition, and no alternative URMA
or HCCP protocol.
