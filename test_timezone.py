#!/usr/bin/env python3
"""테스트: timezone 설정 검증"""

from app.config import settings
from app.core.central_client import now_iso as central_now_iso
from app.core.task_manager import now_iso as task_now_iso
from app.main import now_iso as main_now_iso

print("=" * 60)
print("Timezone Configuration Test")
print("=" * 60)
print(f"\nSettings app_timezone: {settings.app_timezone}")

print("\n" + "-" * 60)
print("Testing now_iso() functions:")
print("-" * 60)

# Test central_client.now_iso()
ts_central = central_now_iso()
print(f"\n1. central_client.now_iso():")
print(f"   Output: {ts_central}")
print(f"   Has +09:00: {'+09:00' in ts_central}")

# Test task_manager.now_iso()
ts_task = task_now_iso()
print(f"\n2. task_manager.now_iso():")
print(f"   Output: {ts_task}")
print(f"   Has +09:00: {'+09:00' in ts_task}")

# Test main.now_iso()
ts_main = main_now_iso()
print(f"\n3. main.now_iso():")
print(f"   Output: {ts_main}")
print(f"   Has +09:00: {'+09:00' in ts_main}")

print("\n" + "=" * 60)
all_have_offset = all('+09:00' in ts for ts in [ts_central, ts_task, ts_main])
if all_have_offset:
    print("✓ All now_iso() functions return KST timestamps (+09:00)")
else:
    print("✗ Some now_iso() functions don't have +09:00 offset")
print("=" * 60)
