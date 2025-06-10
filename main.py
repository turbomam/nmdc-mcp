from typing import Any, Dict, List, Optional, Union
from fastmcp import FastMCP
import requests

mcp = FastMCP("nmdc_fetcher", description="Fetch NMDC records in a paginated manner")

def fetch_nmdc_biosample_records_paged(
        max_page_size: int = 100,
        projection: Optional[Union[str, List[str]]] = None,
        page_token: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,  # Placeholder for future filtering support
        additional_params: Optional[Dict[str, Any]] = None,
        max_records: Optional[int] = None,
        verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch records from the NMDC API collection in a paging manner.

    Parameters:
        max_page_size (int): Maximum number of records per page.
        projection (str or list of str, optional): Fields to include; either comma-separated or a list.
        page_token (str, optional): Page token from the previous response.
        filter_criteria (dict, optional): MongoDB-like query filter. (Placeholder)
        additional_params (dict, optional): Additional query parameters.
        max_records (int, optional): Maximum total number of records to fetch.
        verbose (bool): If True, print progress and details.

    Returns:
        List[Dict[str, Any]]: Aggregated list of fetched records.
    """
    base_url: str = "https://api.microbiomedata.org/nmdcschema"
    collection: str = "biosample_set"

    all_records = []
    endpoint_url = f"{base_url}/{collection}"
    params = {
        "max_page_size": max_page_size
    }

    if projection:
        if isinstance(projection, list):
            params["projection"] = ",".join(projection)
        else:
            params["projection"] = projection

    if page_token:
        params["page_token"] = page_token

    # Placeholder: Filter criteria would normally be serialized to JSON and added here.
    if filter_criteria:
        # params["filter"] = json.dumps(filter_criteria)
        pass

    if additional_params:
        params.update(additional_params)

    while True:
        response = requests.get(endpoint_url, params=params)
        response.raise_for_status()
        data = response.json()

        records = data.get("resources", [])
        all_records.extend(records)

        if verbose:
            print(f"Fetched {len(records)} records; total so far: {len(all_records)}")

        # Check if we've hit the max_records limit
        if max_records is not None and len(all_records) >= max_records:
            all_records = all_records[:max_records]
            if verbose:
                print(f"Reached max_records limit: {max_records}. Stopping fetch.")
            break

        next_page_token = data.get("next_page_token")
        if next_page_token:
            params["page_token"] = next_page_token
        else:
            break

    return all_records

@mcp.tool
def get_samples_above_elevation(min_elevation: int, max_elevation) -> List[Dict[str, Any]]:
    """
    Fetch NMDC biosample records with elevation within a specified range.

    Args:
        min_elevation (int): Minimum elevation (exclusive) for filtering records.
        max_elevation (int): Maximum elevation (exclusive) for filtering records.

    Returns:
        List[Dict[str, Any]]: List of biosample records that have elevation greater than min_elevation and less than max_elevation.
    """
    filter_criteria = {
        "elev": {"$gt": min_elevation, "$lt": max_elevation}
    }
    
    records = fetch_nmdc_biosample_records_paged(
        filter_criteria=filter_criteria,
        max_records=10,
    )
    
    return records

if __name__ == "__main__":
    mcp.run()