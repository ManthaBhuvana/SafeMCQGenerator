import sys
import types
import streamlit.watcher.local_sources_watcher as watcher

# Create a safe version of extract_module_paths
original_extract_module_paths = watcher.extract_module_paths

def safe_extract_module_paths(module):
    # Skip torch._classes module which causes problems
    if hasattr(module, "__name__") and (module.__name__ == "torch._classes" or module.__name__.startswith("torch._classes.")):
        return []
    
    try:
        return original_extract_module_paths(module)
    except Exception:
        return []

# Apply the patch
watcher.extract_module_paths = safe_extract_module_paths
