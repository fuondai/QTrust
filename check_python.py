import sys
import pkg_resources

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\nInstalled packages:")
for package in pkg_resources.working_set:
    print(f"{package.key} {package.version}") 