import argparse
import sys
from typing import List, Optional
from aim import Repo

from aim.sdk.sequence_collection import QueryReportMode, QueryRunSequenceCollection

def fetch_dvc_exp_ids(
    query: str,
    repo_path: str = ".",
    dvc_id_param_key: str = "dvc_exp_id",
    verbose: bool = True
) -> List[str]:
    """
    Queries an Aim repository and retrieves DVC experiment IDs matching the query.

    Args:
        query (str): Aim search query (e.g., "run.params.dataset == 'A'").
        repo_path (str): Path to the Aim repository (.aim folder location).
        dvc_id_param_key (str): Key where dvc_exp_id is stored in run params or context.

    Returns:
        List[str]: Unique list of DVC experiment IDs or hashes matching the criteria.
    """
    repo = Repo(repo_path)
    
    runs: QueryRunSequenceCollection = repo.query_runs(
        query=query, report_mode=QueryReportMode.DISABLED
    )
    df = runs.dataframe()

    mask = df[dvc_id_param_key].notnull()
    runs = df[mask][dvc_id_param_key]
    dvc_exp_ids = runs.tolist()

    mask = df[dvc_id_param_key].isnull()
    runs = df[mask]["hash"]

    if verbose:
        num_duplicates = len(dvc_exp_ids) - len(set(dvc_exp_ids))
        if num_duplicates > 0:
            print(f"Warning: Found {num_duplicates} duplicated DVC experiment IDs in the query results:")
            duplicates = [x for x in dvc_exp_ids if dvc_exp_ids.count(x) > 1]
            print(f"Duplicated DVC experiment IDs: {set(duplicates)}")

        if not runs.empty:
            print(f"Warning: The following {len(runs)} runs do not have a '{dvc_id_param_key}' parameter:")
            for run_hash in runs:
                print(f"{run_hash}")

    return list(set(dvc_exp_ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query AimStack repo and extract corresponding DVC Experiment IDs."
    )
    parser.add_argument(
        "-q", "--query", 
        type=str, 
        required=True, 
        help="Aim query string (e.g., \"run.params.dataset == 'A' or run.params.model == 'F'\")"
    )
    parser.add_argument(
        "-r", "--repo", 
        type=str, 
        default=".", 
        help="Path to .aim repository directory (default: current directory)"
    )
    parser.add_argument(
        "-k", "--key", 
        type=str, 
        default="dvc_exp_id", 
        help="Param key under which the DVC Exp ID is logged (default: dvc_exp_id)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Print warnings for runs without DVC Exp ID or duplicated IDs"
    )

    args = parser.parse_args()

    matching_ids = fetch_dvc_exp_ids(
        query=args.query, 
        repo_path=args.repo, 
        dvc_id_param_key=args.key,
        verbose=args.verbose
    )

    if matching_ids:
        # Print space-separated list ready to pipe into dvc exp remove / xargs
        print(" ".join(matching_ids))
    else:
        sys.exit(0)