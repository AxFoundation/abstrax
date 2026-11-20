import socket
import sys
from collections import defaultdict
from importlib import import_module
from platform import python_version

import pandas as pd
import strax

__all__ = ["print_versions"]

_is_jupyter = any("jupyter" in arg for arg in sys.argv)


def print_versions(
    modules=("strax", "abstrax"),
    print_output=not _is_jupyter,
    include_python=True,
    return_string=False,
    include_git=True,
):
    """Print versions of modules installed.

    :param modules: Modules to print, should be str, tuple or list. E.g.
        print_versions(modules=('numpy', 'dddm',))
    :param print_output: print the version table
    :param include_python: include the python version and executable
    :param return_string: optional. Instead of returning the dataframe, return a string
    :param include_git: Include the current branch and latest commit hash (requires GitPython)
    :return: the message (if return_string) or a dataframe with the versions

    """
    versions = defaultdict(list)
    if include_python:
        versions["module"] = ["python"]
        versions["version"] = [python_version()]
        versions["path"] = [sys.executable]
        versions["git"] = [None]
    for m in strax.to_str_tuple(modules):
        result = _version_info_for_module(m, include_git=include_git)
        if result is None:
            continue
        version, path, git_info = result
        versions["module"].append(m)
        versions["version"].append(version)
        versions["path"].append(path)
        versions["git"].append(git_info)
    df = pd.DataFrame(versions)
    info = f"Host {socket.getfqdn()}\n{df.to_string(index=False,)}"
    if print_output:
        print(info)
    if return_string:
        return info
    return df


def _version_info_for_module(module_name, include_git):
    try:
        mod = import_module(module_name)
    except (ModuleNotFoundError, ImportError):
        print(f"{module_name} is not installed")
        return
    git = None
    version = mod.__dict__.get("__version__", None)
    module_path = mod.__dict__.get("__path__", [None])[0]
    if include_git and module_path is not None:
        git = _git_info(module_path)
    return version, module_path, git


def _git_info(path):
    try:
        from git import InvalidGitRepositoryError, NoSuchPathError, Repo
    except ImportError:
        return None
    try:
        repo = Repo(path, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return None
    try:
        branch = repo.active_branch
    except TypeError:
        branch = "unknown"
    try:
        commit_hash = repo.head.object.hexsha
    except (TypeError, ValueError):
        commit_hash = "unknown"
    return f"branch:{branch} | {commit_hash[:7]}"
