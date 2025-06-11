import json
import pprint
import re
from typing import Any, Dict, List, Optional, Union

import requests
from fastmcp import FastMCP

mcp = FastMCP("nmdc_fetcher", description="Fetch NMDC records in a paginated manner")


# todo: fetch_nmdc_biosample_records_paged and fetch_nmdc_records_paged are highly redundant
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


@mcp.tool
def list_nmdc_collections(
        base_url: str = "https://api.microbiomedata.org/nmdcschema",
        verbose: bool = False,
) -> List[str]:
    """
    Retrieve all valid NMDC Schema collection names.

    This function returns the names of all available collections in the NMDC database,
    corresponding to the slots of the nmdc:Database class that describe database collections.

    Parameters:
        base_url (str): Base URL of the NMDC API (excluding specific endpoints).
        verbose (bool): If True, print additional details about the request.

    Returns:
        List[str]: List of collection names available in the NMDC database.
    """
    endpoint_url = f"{base_url}/collection_names"

    if verbose:
        print(f"Fetching collection names from: {endpoint_url}")

    try:
        response = requests.get(endpoint_url)
        response.raise_for_status()
        collections = response.json()

        if verbose:
            print(f"Successfully retrieved {len(collections)} collections")
            pprint.pprint(collections)

        return collections

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch collection names: {str(e)}")


@mcp.tool
def fetch_nmdc_records_paged(
        base_url: str = "https://api.microbiomedata.org/nmdcschema",
        collection: str = "biosample_set",
        filter_criteria: Optional[Dict[str, Any]] = None,
        max_records: Optional[int] = None,
        projection: Optional[Union[str, List[str]]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        max_page_size: int = 100,
        page_token: Optional[str] = None,
        verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch records from the NMDC API collection in a paging manner.

    Parameters:
        base_url (str): Base URL of the NMDC API (excluding collection name).
        collection (str): Collection name to fetch (e.g., 'biosample_set').
        filter_criteria (dict, optional): MongoDB-like query filter.
        max_records (int, optional): Maximum total number of records to fetch.
        projection (str or list of str, optional): Fields to include; either comma-separated or a list.
        additional_params (dict, optional): Additional query parameters.
        max_page_size (int): Maximum number of records per page.
        page_token (str, optional): Page token from the previous response
        verbose (bool): If True, print progress and details.

    Returns:
        List[Dict[str, Any]]: Aggregated list of fetched records.
    """
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
        params["filter"] = json.dumps(filter_criteria)
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
def examine_nmdc_schema_classes(
        schema_url: str = "https://microbiomedata.github.io/nmdc-schema/#classes",
        include_properties: bool = True,
        include_inheritance: bool = True,
        max_classes: Optional[int] = None,
        focus_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Examine and analyze NMDC schema classes from the documentation to understand the data model.

    This function scrapes the NMDC schema documentation to extract information about
    classes, their properties, inheritance relationships, and categorization.

    Parameters:
        schema_url (str): URL to the NMDC schema classes documentation
        include_properties (bool): Whether to extract property information for classes
        include_inheritance (bool): Whether to analyze inheritance relationships
        max_classes (int, optional): Maximum number of classes to analyze
        focus_categories (List[str], optional): Specific categories to focus on

    Returns:
        Dict[str, Any]: Comprehensive analysis of NMDC schema classes
    """
    try:
        # Import BeautifulSoup for HTML parsing
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "BeautifulSoup4 is required. Install with: pip install beautifulsoup4"}

        # Fetch the schema documentation page
        response = requests.get(schema_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        analysis = {
            "schema_info": extract_schema_metadata(soup),
            "classes": analyze_schema_classes(soup, include_properties, max_classes),
            "categories": categorize_schema_classes(soup),
            "summary": generate_schema_summary(soup)
        }

        if include_inheritance:
            analysis["inheritance"] = analyze_class_inheritance(soup)

        if focus_categories:
            analysis["focused_analysis"] = focus_on_categories(analysis, focus_categories)

        return analysis

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch schema documentation: {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing schema classes: {str(e)}"}


def extract_schema_metadata(soup) -> Dict[str, Any]:
    """Extract basic metadata about the NMDC schema."""
    metadata = {
        "title": "NMDC Schema Classes",
        "description": "Unknown"
    }

    # Try to find title and description
    title_elem = soup.find('h1') or soup.find('title')
    if title_elem:
        metadata["title"] = title_elem.get_text().strip()

    # Look for schema description
    desc_elem = soup.find('p', class_='description') or soup.find('div', class_='description')
    if desc_elem:
        metadata["description"] = desc_elem.get_text().strip()[:200]

    return metadata


def analyze_schema_classes(soup, include_properties: bool = True, max_classes: Optional[int] = None) -> Dict[str, Any]:
    """Analyze individual schema classes from the documentation."""
    classes = {}

    # First, try to find explicit class definitions
    class_sections = []

    # Try different selectors for class information
    class_sections.extend(soup.find_all('div', class_='class'))
    class_sections.extend(soup.find_all('section', id=lambda x: x and 'class-' in x))

    # Look for elements with "class" in the ID
    class_sections.extend(soup.find_all(id=lambda x: x and 'class' in x.lower()))

    # If we don't find explicit class sections, extract class names from text
    if len(class_sections) < 10:  # If we have few explicit sections
        classes.update(extract_classes_from_text(soup))

    # Look for headings that might be class names (start with capital letter)
    headings = soup.find_all(['h2', 'h3', 'h4'])
    for heading in headings:
        text = heading.get_text().strip()
        # Check if it looks like a class name (starts with capital, has mix of cases)
        if text and text[0].isupper() and any(char.islower() for char in text) and len(text) > 3:
            # Avoid navigation and common headers
            if not any(word in text.lower() for word in ['class', 'type', 'schema', 'overview', 'index', 'navigation']):
                class_sections.append(heading)

    count = len(classes)  # Start count from text-extracted classes
    for section in class_sections:
        if max_classes and count >= max_classes:
            break

        class_info = extract_class_info(section, soup, include_properties)
        if class_info and class_info.get('name') and len(class_info['name']) > 2:
            if class_info['name'] not in classes:  # Avoid duplicates
                classes[class_info['name']] = class_info
                count += 1

    return {
        "total_found": len(classes),
        "classes": classes
    }


def extract_classes_from_text(soup) -> Dict[str, Dict[str, Any]]:
    """Extract class names and information from the page text content using dynamic detection."""
    classes = {}
    text_content = soup.get_text()

    # Look for CamelCase words that might be classes (dynamic detection)
    camel_case_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b'
    camel_case_words = re.findall(camel_case_pattern, text_content)

    # Count occurrences to find important terms
    word_counts = {}
    for word in camel_case_words:
        if len(word) > 4:  # Only consider words longer than 4 characters
            word_counts[word] = word_counts.get(word, 0) + 1

    # Consider words that appear multiple times or match common patterns
    for word, count in word_counts.items():
        if (count > 1 or  # Appears multiple times
                word.endswith(('Set', 'Type', 'Enum', 'Value', 'Info', 'Data')) or  # Common suffixes
                any(pattern in word.lower() for pattern in
                    ['sample', 'data', 'workflow', 'annotation', 'study', 'process', 'object'])):
            class_info = {
                "name": word,
                "description": extract_class_description_from_context(soup, word),
                "properties": [],
                "parent_class": None,
                "category": identify_class_category(word, ""),
                "occurrence_count": count
            }
            classes[word] = class_info

    return classes


def extract_class_description_from_context(soup, class_name: str) -> str:
    """Try to extract description for a class by finding it in context."""
    # Look for the class name in text and try to get surrounding context
    text_content = soup.get_text()

    # Find sentences containing the class name
    sentences = text_content.split('.')
    for sentence in sentences:
        if class_name in sentence:
            # Clean up and return the sentence as description
            clean_sentence = sentence.strip()
            if len(clean_sentence) > 20 and len(clean_sentence) < 300:
                return clean_sentence

    # If no sentence found, look in nearby elements
    all_text_elements = soup.find_all(string=lambda text: class_name in text if text else False)
    for element in all_text_elements[:1]:  # Just check first occurrence
        parent = element.parent
        if parent:
            text = parent.get_text().strip()
            # if len(text) > 20 and len(text) < 300:
            #     return text[:200] + "..." if len(text) > 200 else text
            return text

    return "No description available"


def extract_class_info(section, soup, include_properties: bool) -> Dict[str, Any]:
    """Extract information about a single class."""
    class_info = {
        "name": "",
        "description": "",
        "properties": [],
        "parent_class": None,
        "category": "unknown"
    }

    # Extract class name
    if section.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        class_info["name"] = section.get_text().strip()
    else:
        name_elem = section.find(['h1', 'h2', 'h3', 'h4']) or section.find(class_='class-name')
        if name_elem:
            class_info["name"] = name_elem.get_text().strip()

    # Extract description
    desc_elem = section.find_next('p') or section.find('div', class_='description')
    if desc_elem:
        # class_info["description"] = desc_elem.get_text().strip()[:300]
        class_info["description"] = desc_elem.get_text().strip()

    # Extract properties if requested
    if include_properties:
        class_info["properties"] = extract_class_properties(section, soup)

    # Try to identify category/type
    class_info["category"] = identify_class_category(class_info["name"], class_info["description"])

    return class_info


def extract_class_properties(section, soup) -> List[Dict[str, Any]]:
    """Extract properties/attributes for a class."""
    properties = []

    # Look for property tables or lists near the class definition
    prop_containers = []

    # Find sibling elements that might contain properties
    current = section
    for _ in range(5):  # Look at next 5 siblings
        current = current.find_next_sibling()
        if not current:
            break
        if current.name in ['table', 'ul', 'ol', 'div']:
            prop_containers.append(current)

    for container in prop_containers:
        if container.name == 'table':
            # Extract from table
            rows = container.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    prop_name = cells[0].get_text().strip()
                    prop_desc = cells[1].get_text().strip()
                    if prop_name and not prop_name.lower() in ['property', 'attribute', 'name']:
                        properties.append({
                            "name": prop_name,
                            "description": prop_desc[:150],
                            "type": extract_property_type(cells)
                        })

        elif container.name in ['ul', 'ol']:
            # Extract from list
            items = container.find_all('li')
            for item in items:
                text = item.get_text().strip()
                if ':' in text:
                    name, desc = text.split(':', 1)
                    properties.append({
                        "name": name.strip(),
                        "description": desc.strip()[:150],
                        "type": "unknown"
                    })

    # return properties[:10]  # Limit to first 10 properties
    return properties  # Limit to first 10 properties


def extract_property_type(cells) -> str:
    """Try to extract property type from table cells."""
    if len(cells) >= 3:
        type_text = cells[2].get_text().strip().lower()
        for type_keyword in ['string', 'integer', 'boolean', 'array', 'object', 'uri', 'date']:
            if type_keyword in type_text:
                return type_keyword
    return "unknown"


def identify_class_category(name: str, description: str) -> str:
    """Identify the category/type of a class based on name and description using dynamic patterns."""
    name_lower = name.lower()
    desc_lower = description.lower()

    # Dynamic pattern detection - build patterns from common word roots
    bio_patterns = [word for word in ['sample', 'specimen'] if word in name_lower or word in desc_lower]
    study_patterns = [word for word in ['study', 'project', 'investigation'] if
                      word in name_lower or word in desc_lower]
    data_patterns = [word for word in ['data', 'file', 'object'] if word in name_lower or word in desc_lower]
    workflow_patterns = [word for word in ['workflow', 'execution', 'activity', 'process'] if
                         word in name_lower or word in desc_lower]
    annotation_patterns = [word for word in ['annotation', 'feature'] if word in name_lower or word in desc_lower]
    person_patterns = [word for word in ['person', 'user', 'contact'] if word in name_lower or word in desc_lower]
    instrument_patterns = [word for word in ['instrument', 'device', 'equipment'] if
                           word in name_lower or word in desc_lower]

    # Return category based on which patterns match
    if bio_patterns or 'biosample' in name_lower:
        return "biological_sample"
    elif study_patterns:
        return "study"
    elif 'organism' in name_lower or 'taxa' in name_lower or 'species' in name_lower:
        return "organism"
    elif data_patterns:
        return "data_object"
    elif workflow_patterns:
        return "workflow"
    elif annotation_patterns:
        return "annotation"
    elif person_patterns:
        return "person"
    elif instrument_patterns:
        return "instrument"
    elif 'protocol' in name_lower or 'method' in name_lower or 'procedure' in name_lower:
        return "protocol"

    return "unknown"


def categorize_schema_classes(soup) -> Dict[str, Any]:
    """Categorize classes by their apparent function/domain using dynamic discovery."""
    # Build categories dynamically from discovered classes
    categories = {}
    category_counts = {}

    # The actual categorization happens in analyze_schema_classes
    # This function provides the structure and would be populated by that function
    return {
        "categories": categories,
        "category_counts": category_counts,
        "note": "Categories are populated dynamically during class analysis"
    }


def analyze_class_inheritance(soup) -> Dict[str, Any]:
    """Analyze inheritance relationships between classes."""
    inheritance = {
        "parent_child_relationships": {},
        "inheritance_depth": {},
        "root_classes": [],
        "leaf_classes": []
    }

    # Look for inheritance keywords in class descriptions
    class_sections = soup.find_all(['h2', 'h3', 'h4'])

    for section in class_sections:
        class_name = section.get_text().strip()

        # Look for inheritance indicators in nearby text
        next_content = section.find_next(['p', 'div'])
        if next_content:
            text = next_content.get_text().lower()
            if any(keyword in text for keyword in ['inherits', 'extends', 'subclass', 'parent']):
                # Try to extract parent class name
                for word in text.split():
                    if word.istitle() and len(word) > 3:
                        inheritance["parent_child_relationships"][class_name] = word
                        break

    return inheritance


def focus_on_categories(analysis: Dict[str, Any], focus_categories: List[str]) -> Dict[str, Any]:
    """Focus analysis on specific categories of classes."""
    focused = {
        "requested_categories": focus_categories,
        "found_classes": {}
    }

    classes = analysis.get("classes", {}).get("classes", {})
    for class_name, class_info in classes.items():
        if class_info.get("category") in focus_categories:
            focused["found_classes"][class_name] = class_info

    return focused


def generate_schema_summary(soup) -> Dict[str, Any]:
    """Generate a high-level summary of the schema using dynamic analysis."""
    # Count different types of elements
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    tables = soup.find_all('table')
    lists = soup.find_all(['ul', 'ol'])

    # Dynamically discover key terms from the content
    text_content = soup.get_text().lower()

    # Use term frequency to identify important concepts
    common_schema_terms = ['class', 'property', 'attribute', 'type', 'object', 'schema']
    domain_terms = ['sample', 'study', 'workflow', 'data', 'annotation', 'analysis']

    found_schema_terms = [term for term in common_schema_terms if term in text_content]
    found_domain_terms = [term for term in domain_terms if term in text_content]

    return {
        "structure_elements": {
            "headings": len(headings),
            "tables": len(tables),
            "lists": len(lists)
        },
        "schema_terms_found": found_schema_terms,
        "domain_terms_found": found_domain_terms,
        "appears_comprehensive": len(found_schema_terms) >= 3 and len(found_domain_terms) >= 3,
        "estimated_complexity": "high" if len(headings) > 20 else "medium" if len(headings) > 10 else "low"
    }


@mcp.tool
def analyze_collection_class_mappings(
        database_url: str = "https://microbiomedata.github.io/nmdc-schema/Database/",
        classes_url: str = "https://microbiomedata.github.io/nmdc-schema/#classes",
        verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze which class instances are stored in each NMDC collection by examining Database slots.

    This function maps collection names to the types of class instances they contain by:
    1. Analyzing Database class slots (which correspond to collection names)
    2. Extracting the range/type information for each slot
    3. Attempting to identify subclass relationships from the classes documentation
    4. Providing a mapping of collections to their instance types

    Parameters:
        database_url (str): URL to the Database class documentation
        classes_url (str): URL to the classes overview page for subclass analysis
        verbose (bool): If True, print detailed analysis information

    Returns:
        Dict[str, Any]: Analysis of collection-to-class mappings including subclass hints
    """
    try:
        # Import BeautifulSoup for HTML parsing
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "BeautifulSoup4 is required. Install with: pip install beautifulsoup4"}

        analysis = {
            "database_info": {},
            "collection_mappings": {},
            "subclass_hints": {},
            "summary": {}
        }

        # Analyze Database class slots
        if verbose:
            print(f"Fetching Database class information from: {database_url}")

        db_response = requests.get(database_url)
        db_response.raise_for_status()
        db_soup = BeautifulSoup(db_response.content, 'html.parser')

        analysis["database_info"] = extract_database_metadata(db_soup)
        analysis["collection_mappings"] = extract_database_slots(db_soup, verbose)

        # Analyze class hierarchy for subclass relationships
        if verbose:
            print(f"Fetching class hierarchy information from: {classes_url}")

        classes_response = requests.get(classes_url)
        classes_response.raise_for_status()
        classes_soup = BeautifulSoup(classes_response.content, 'html.parser')

        analysis["subclass_hints"] = analyze_class_hierarchy(classes_soup, verbose)

        # Generate summary
        analysis["summary"] = generate_collection_summary(analysis)

        return analysis

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch documentation: {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing collection mappings: {str(e)}"}


def extract_database_metadata(soup) -> Dict[str, Any]:
    """Extract metadata about the Database class."""
    metadata = {
        "class_name": "Database",
        "description": "Unknown",
        "url_analyzed": True
    }

    # Look for class description
    desc_elem = soup.find('p') or soup.find('div', class_='description')
    if desc_elem:
        metadata["description"] = desc_elem.get_text().strip()[:300]

    # Look for class title/heading
    title_elem = soup.find('h1') or soup.find('h2')
    if title_elem:
        title_text = title_elem.get_text().strip()
        if 'Database' in title_text:
            metadata["class_name"] = title_text

    return metadata


def extract_database_slots(soup, verbose: bool = False) -> Dict[str, Any]:
    """Extract Database class slots and their ranges (collection mappings)."""
    slots = {}

    # Look for slots/properties table or section
    tables = soup.find_all('table')
    slot_containers = soup.find_all(['table', 'div', 'section'])

    if verbose:
        print(f"Found {len(tables)} tables to analyze for slots")

    # Try different approaches to find slot information
    for container in slot_containers:
        container_slots = extract_slots_from_container(container, verbose)
        slots.update(container_slots)

    # Also look for individual slot definitions
    slot_links = soup.find_all('a', href=lambda x: x and '/slots/' in x)
    for link in slot_links:
        slot_name = link.get_text().strip()
        if slot_name and '_set' in slot_name:  # Likely a collection slot
            slot_info = extract_slot_details_from_context(soup, link, verbose)
            if slot_info:
                slots[slot_name] = slot_info

    if verbose:
        print(f"Extracted {len(slots)} slots total")

    return {
        "total_slots": len(slots),
        "collection_slots": {k: v for k, v in slots.items() if k.endswith('_set')},
        "all_slots": slots
    }


def extract_slots_from_container(container, verbose: bool = False) -> Dict[str, Any]:
    """Extract slot information from a table or container."""
    slots = {}

    if container.name == 'table':
        # Extract from table format
        rows = container.find_all('tr')
        headers = []

        # Get headers
        if rows:
            header_row = rows[0]
            headers = [th.get_text().strip().lower() for th in header_row.find_all(['th', 'td'])]

        # Find relevant columns
        name_col = -1
        range_col = -1
        desc_col = -1

        for i, header in enumerate(headers):
            if 'name' in header or 'slot' in header:
                name_col = i
            elif 'range' in header or 'type' in header:
                range_col = i
            elif 'description' in header or 'desc' in header:
                desc_col = i

        # Extract slot data
        for row in rows[1:]:  # Skip header
            cells = row.find_all(['td', 'th'])
            if len(cells) > max(name_col, range_col, desc_col):
                slot_name = cells[name_col].get_text().strip() if name_col >= 0 else ""
                slot_range = cells[range_col].get_text().strip() if range_col >= 0 else ""
                slot_desc = cells[desc_col].get_text().strip() if desc_col >= 0 else ""

                if slot_name and (slot_name.endswith('_set') or 'set' in slot_name.lower()):
                    slots[slot_name] = {
                        "range": slot_range,
                        "description": slot_desc[:200],
                        "is_collection": True,
                        "source": "table_extraction"
                    }

                    if verbose:
                        print(f"Found collection slot: {slot_name} -> {slot_range}")

    return slots


def extract_slot_details_from_context(soup, link_element, verbose: bool = False) -> Dict[str, Any]:
    """Extract slot details from context around a slot link."""
    slot_info = {
        "range": "Unknown",
        "description": "",
        "is_collection": False,
        "source": "context_extraction"
    }

    # Look for range information near the link
    parent = link_element.parent
    if parent:
        # Look for range patterns in nearby text
        nearby_text = parent.get_text()

        # Look for "range:" or similar patterns
        if 'range:' in nearby_text.lower():
            range_match = re.search(r'range:\s*([A-Za-z_][A-Za-z0-9_]*)', nearby_text, re.IGNORECASE)
            if range_match:
                slot_info["range"] = range_match.group(1)

        # Look for class names (CamelCase words)
        camel_case_words = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', nearby_text)
        if camel_case_words:
            # Take the first substantial CamelCase word as potential range
            for word in camel_case_words:
                if len(word) > 3 and word not in ['Database', 'Class', 'Slot']:
                    slot_info["range"] = word
                    break

        slot_info["description"] = nearby_text.strip()[:200]

    slot_name = link_element.get_text().strip()
    slot_info["is_collection"] = slot_name.endswith('_set')

    return slot_info


def analyze_class_hierarchy(soup, verbose: bool = False) -> Dict[str, Any]:
    """Analyze class hierarchy to identify potential subclass relationships."""
    hierarchy = {
        "indentation_analysis": {},
        "parent_child_hints": {},
        "class_levels": {},
        "potential_subclasses": {}
    }

    # Look for the classes table or list
    tables = soup.find_all('table')

    for table in tables:
        # Check if this looks like a classes table
        headers = table.find_all('th') or table.find_all('td')[:1]
        if headers and any('class' in h.get_text().lower() for h in headers):
            hierarchy_info = extract_hierarchy_from_table(table, verbose)
            hierarchy.update(hierarchy_info)
            break

    # Also look for nested lists that might indicate hierarchy
    lists = soup.find_all(['ul', 'ol'])
    for lst in lists:
        if len(lst.find_all('li')) > 10:  # Likely a substantial list
            list_hierarchy = extract_hierarchy_from_list(lst, verbose)
            if list_hierarchy:
                hierarchy["list_based_hierarchy"] = list_hierarchy

    return hierarchy


def extract_hierarchy_from_table(table, verbose: bool = False) -> Dict[str, Any]:
    """Extract class hierarchy information from a table using indentation analysis."""
    hierarchy = {
        "indentation_analysis": {},
        "class_levels": {},
        "potential_subclasses": {}
    }

    rows = table.find_all('tr')

    for row in rows[1:]:  # Skip header
        cells = row.find_all(['td', 'th'])
        if cells:
            first_cell = cells[0]
            class_name_elem = first_cell.find('a') or first_cell

            if class_name_elem:
                class_name = class_name_elem.get_text().strip()

                # Analyze indentation by looking at CSS classes, styles, or nested elements
                indentation_level = analyze_indentation(first_cell)

                # Look for visual hierarchy hints
                if class_name and len(class_name) > 2:
                    hierarchy["class_levels"][class_name] = indentation_level

                    # If this appears indented, try to find its parent
                    if indentation_level > 0:
                        parent_class = find_likely_parent_class(hierarchy["class_levels"], indentation_level)
                        if parent_class:
                            if parent_class not in hierarchy["potential_subclasses"]:
                                hierarchy["potential_subclasses"][parent_class] = []
                            hierarchy["potential_subclasses"][parent_class].append(class_name)

                            if verbose:
                                print(f"Potential subclass relationship: {class_name} -> {parent_class}")

    return hierarchy


def analyze_indentation(cell_element) -> int:
    """Analyze the indentation level of a table cell."""
    indentation = 0

    # Look for CSS classes that might indicate indentation
    css_classes = cell_element.get('class', [])
    for css_class in css_classes:
        if 'indent' in css_class.lower():
            # Try to extract indentation level from class name
            level_match = re.search(r'(\d+)', css_class)
            if level_match:
                indentation = int(level_match.group(1))

    # Look for style attributes
    style = cell_element.get('style', '')
    if 'margin-left' in style or 'padding-left' in style:
        # Try to extract pixel values and convert to levels
        pixel_match = re.search(r'(\d+)px', style)
        if pixel_match:
            pixels = int(pixel_match.group(1))
            indentation = pixels // 20  # Assume ~20px per level

    # Look for nested elements that might indicate indentation
    nested_divs = len(cell_element.find_all('div'))
    if nested_divs > 1:
        indentation = max(indentation, nested_divs - 1)

    # Look for non-breaking spaces or other spacing
    text_content = cell_element.get_text()
    leading_spaces = len(text_content) - len(text_content.lstrip(' \t\xa0'))
    if leading_spaces > 0:
        indentation = max(indentation, leading_spaces // 4)

    return indentation


def find_likely_parent_class(class_levels: Dict[str, int], current_level: int) -> str:
    """Find the most likely parent class for a given indentation level."""
    # Look for the most recent class at a lower indentation level
    candidates = []
    for class_name, level in class_levels.items():
        if level == current_level - 1:
            candidates.append(class_name)

    # Return the last one found (most recent in document order)
    return candidates[-1] if candidates else None


def extract_hierarchy_from_list(lst, verbose: bool = False) -> Dict[str, Any]:
    """Extract hierarchy from nested lists."""
    hierarchy = {}

    def process_list_items(items, level=0, parent=None):
        for item in items:
            text = item.get_text(strip=True)
            if text and len(text) > 2:
                # Look for class-like names
                if re.match(r'^[A-Z][a-zA-Z0-9]*$', text.split()[0]):
                    class_name = text.split()[0]
                    hierarchy[class_name] = {
                        "level": level,
                        "parent": parent,
                        "full_text": text
                    }

                    # Process nested lists
                    nested_lists = item.find_all(['ul', 'ol'], recursive=False)
                    for nested_list in nested_lists:
                        nested_items = nested_list.find_all('li', recursive=False)
                        process_list_items(nested_items, level + 1, class_name)

    items = lst.find_all('li', recursive=False)
    process_list_items(items)

    return hierarchy


def generate_collection_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the collection-class mapping analysis."""
    collection_mappings = analysis.get("collection_mappings", {})
    collection_slots = collection_mappings.get("collection_slots", {})
    subclass_hints = analysis.get("subclass_hints", {})

    summary = {
        "total_collections": len(collection_slots),
        "collections_with_ranges": len([s for s in collection_slots.values() if s.get("range") != "Unknown"]),
        "potential_subclass_relationships": len(subclass_hints.get("potential_subclasses", {})),
        "collection_overview": {},
        "recommendations": []
    }

    # Create overview of each collection
    for collection_name, slot_info in collection_slots.items():
        base_class = slot_info.get("range", "Unknown")
        potential_subclasses = []

        # Look for potential subclasses
        if base_class in subclass_hints.get("potential_subclasses", {}):
            potential_subclasses = subclass_hints["potential_subclasses"][base_class]

        summary["collection_overview"][collection_name] = {
            "primary_class": base_class,
            "potential_subclasses": potential_subclasses,
            "total_possible_types": 1 + len(potential_subclasses),
            "description": slot_info.get("description", "")[:100]
        }

    # Generate recommendations
    if summary["collections_with_ranges"] < summary["total_collections"]:
        summary["recommendations"].append("Some collections missing range information - may need manual verification")

    if summary["potential_subclass_relationships"] > 0:
        summary["recommendations"].append(
            f"Found {summary['potential_subclass_relationships']} potential subclass relationships - collections may contain more types than just their primary class")

    return summary


@mcp.tool
def analyze_class_slots(
        class_name: str,
        schema_base_url: str = "https://microbiomedata.github.io/nmdc-schema/",
        verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze the slots (properties/attributes) of a specific NMDC schema class.

    This function fetches the class documentation page and extracts information about
    the class's slots, including their ranges, descriptions, and other metadata.

    Parameters:
        class_name (str): Name of the class to analyze (e.g., 'Study', 'Biosample')
        schema_base_url (str): Base URL of the NMDC schema documentation
        verbose (bool): If True, print detailed analysis information

    Returns:
        Dict[str, Any]: Analysis of the class slots including ranges and descriptions
    """
    try:
        # Import BeautifulSoup for HTML parsing
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "BeautifulSoup4 is required. Install with: pip install beautifulsoup4"}

        # Construct the class URL
        class_url = f"{schema_base_url.rstrip('/')}/{class_name}/"

        if verbose:
            print(f"Fetching class information from: {class_url}")

        # Fetch the class documentation page
        response = requests.get(class_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        analysis = {
            "class_info": extract_class_metadata(soup, class_name),
            "slots": extract_class_slots_table(soup, verbose),
            "inheritance_info": extract_class_inheritance_info(soup),
            "summary": {}
        }

        # Generate summary
        analysis["summary"] = generate_class_slots_summary(analysis)

        if verbose:
            slots_count = len(analysis["slots"])
            print(f"Found {slots_count} slots for class {class_name}")

        return analysis

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch class documentation for '{class_name}': {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing class slots for '{class_name}': {str(e)}"}


def extract_class_metadata(soup, class_name: str) -> Dict[str, Any]:
    """Extract metadata about the class from its documentation page."""
    metadata = {
        "class_name": class_name,
        "description": "No description available",
        "title": "",
        "uri": ""
    }

    # Look for class title/heading
    title_elem = soup.find('h1') or soup.find('h2')
    if title_elem:
        metadata["title"] = title_elem.get_text().strip()

    # Look for class description - usually in first paragraph or description section
    desc_candidates = [
        soup.find('p'),
        soup.find('div', class_='description'),
        soup.find('div', id='description'),
        soup.find('section', class_='description')
    ]

    for candidate in desc_candidates:
        if candidate and candidate.get_text().strip():
            # metadata["description"] = candidate.get_text().strip()[:500]
            metadata["description"] = candidate.get_text().strip()
            break

    # Look for URI/identifier information
    uri_elem = soup.find('code') or soup.find('span', class_='uri')
    if uri_elem and 'nmdc:' in uri_elem.get_text():
        metadata["uri"] = uri_elem.get_text().strip()

    return metadata


def extract_class_slots_table(soup, verbose: bool = False) -> Dict[str, Any]:
    """Extract slots information from the class page's Slots table."""
    slots = {}

    # Look for the "Slots" heading and the table that follows
    slots_heading = None
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

    for heading in headings:
        if 'slot' in heading.get_text().lower():
            slots_heading = heading
            break

    if not slots_heading:
        if verbose:
            print("No 'Slots' heading found, searching for tables...")

    # Find tables, either after the Slots heading or anywhere on the page
    tables = soup.find_all('table')

    if verbose:
        print(f"Found {len(tables)} tables to analyze")

    for table in tables:
        # Check if this looks like a slots table
        if is_slots_table(table):
            slots_data = extract_slots_from_table(table, verbose)
            slots.update(slots_data)
            if verbose:
                print(f"Extracted {len(slots_data)} slots from table")
            break

    return slots


def is_slots_table(table) -> bool:
    """Determine if a table contains slots information."""
    # Look at headers to see if this is a slots table
    headers = table.find_all('th')
    header_texts = [th.get_text().strip().lower() for th in headers]

    # Common patterns for slots tables
    slots_indicators = ['name', 'range', 'domain', 'slot', 'property', 'multivalued', 'required']

    # If we find at least 2 of these indicators, it's likely a slots table
    found_indicators = sum(1 for indicator in slots_indicators if any(indicator in header for header in header_texts))

    return found_indicators >= 2


def extract_slots_from_table(table, verbose: bool = False) -> Dict[str, Any]:
    """Extract slot information from a slots table."""
    slots = {}

    # Get table headers
    rows = table.find_all('tr')
    if not rows:
        return slots

    header_row = rows[0]
    headers = [th.get_text().strip().lower() for th in header_row.find_all(['th', 'td'])]

    if verbose:
        print(f"Table headers: {headers}")

    # Map common column names to indices
    column_map = {}
    for i, header in enumerate(headers):
        if 'name' in header or 'slot' in header:
            column_map['name'] = i
        elif 'range' in header:
            column_map['range'] = i
        elif 'domain' in header:
            column_map['domain'] = i
        elif 'description' in header or 'desc' in header:
            column_map['description'] = i
        elif 'required' in header:
            column_map['required'] = i
        elif 'multivalued' in header or 'multi' in header:
            column_map['multivalued'] = i
        elif 'identifier' in header:
            column_map['identifier'] = i
        elif 'key' in header:
            column_map['key'] = i

    if verbose:
        print(f"Column mapping: {column_map}")

    # Extract slot data from each row
    for row in rows[1:]:  # Skip header row
        cells = row.find_all(['td', 'th'])
        if len(cells) < len(headers):
            continue

        slot_info = extract_slot_info_from_row(cells, column_map, headers)

        if slot_info and slot_info.get('name'):
            slot_name = slot_info['name']
            slots[slot_name] = slot_info

            if verbose:
                print(f"Extracted slot: {slot_name} -> {slot_info.get('range', 'Unknown')}")

    return slots


def extract_slot_info_from_row(cells, column_map: Dict[str, int], headers: List[str]) -> Dict[str, Any]:
    """Extract slot information from a table row."""
    slot_info = {
        "name": "",
        "range": "Unknown",
        "description": "",
        "required": False,
        "multivalued": False,
        "domain": "",
        "identifier": False,
        "key": False
    }

    # Extract information based on column mapping
    for field, col_index in column_map.items():
        if col_index < len(cells):
            cell = cells[col_index]
            cell_text = cell.get_text().strip()

            if field == 'name':
                # Look for link text or plain text
                link = cell.find('a')
                slot_info['name'] = link.get_text().strip() if link else cell_text
            elif field == 'range':
                # Range might be a link to another class
                link = cell.find('a')
                slot_info['range'] = link.get_text().strip() if link else cell_text
            elif field == 'description':
                slot_info['description'] = cell_text[:300]  # Limit description length
            elif field in ['required', 'multivalued', 'identifier', 'key']:
                # Boolean fields
                slot_info[field] = cell_text.lower() in ['true', 'yes', '1', 'required', 'x', '✓']
            else:
                slot_info[field] = cell_text

    # Additional processing for range - look for links that might indicate the type
    range_cell_index = column_map.get('range')
    if range_cell_index is not None and range_cell_index < len(cells):
        range_cell = cells[range_cell_index]
        # Look for multiple links or types
        links = range_cell.find_all('a')
        if len(links) > 1:
            slot_info['range_options'] = [link.get_text().strip() for link in links]
        elif len(links) == 1:
            slot_info['range'] = links[0].get_text().strip()

    return slot_info if slot_info.get('name') else None


def extract_class_inheritance_info(soup) -> Dict[str, Any]:
    """Extract inheritance information for the class."""
    inheritance = {
        "parents": [],
        "children": [],
        "is_a": ""
    }

    # Look for inheritance information in various places
    # Check for "is a" relationships
    text_content = soup.get_text().lower()

    # Look for parent class information
    parent_patterns = [
        r'is[_\s]+a:?\s*([A-Z][a-zA-Z0-9]*)',
        r'inherits?\s+from:?\s*([A-Z][a-zA-Z0-9]*)',
        r'extends:?\s*([A-Z][a-zA-Z0-9]*)',
        r'parent:?\s*([A-Z][a-zA-Z0-9]*)'
    ]

    for pattern in parent_patterns:
        matches = re.findall(pattern, soup.get_text(), re.IGNORECASE)
        if matches:
            inheritance["parents"].extend(matches)
            if matches:
                inheritance["is_a"] = matches[0]  # Take the first/primary parent
            break

    # Look for inheritance info in structured sections
    inheritance_sections = soup.find_all(['div', 'section'], class_=lambda x: x and 'inherit' in x.lower())
    for section in inheritance_sections:
        links = section.find_all('a')
        for link in links:
            class_name = link.get_text().strip()
            if class_name and class_name[0].isupper():
                inheritance["parents"].append(class_name)

    # Remove duplicates
    inheritance["parents"] = list(set(inheritance["parents"]))

    return inheritance


def generate_class_slots_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the class slots analysis."""
    slots = analysis.get("slots", {})
    class_info = analysis.get("class_info", {})
    inheritance = analysis.get("inheritance_info", {})

    # Analyze slot characteristics
    total_slots = len(slots)
    required_slots = sum(1 for slot in slots.values() if slot.get("required", False))
    multivalued_slots = sum(1 for slot in slots.values() if slot.get("multivalued", False))
    identifier_slots = sum(1 for slot in slots.values() if slot.get("identifier", False))

    # Analyze range types
    range_types = {}
    for slot in slots.values():
        range_val = slot.get("range", "Unknown")
        range_types[range_val] = range_types.get(range_val, 0) + 1

    # Categorize slots by range type
    primitive_types = ['string', 'integer', 'boolean', 'float', 'double', 'uri', 'date', 'datetime']
    primitive_slots = sum(1 for slot in slots.values() if slot.get("range", "").lower() in primitive_types)
    class_reference_slots = total_slots - primitive_slots

    summary = {
        "class_name": class_info.get("class_name", "Unknown"),
        "total_slots": total_slots,
        "required_slots": required_slots,
        "optional_slots": total_slots - required_slots,
        "multivalued_slots": multivalued_slots,
        "identifier_slots": identifier_slots,
        "primitive_type_slots": primitive_slots,
        "class_reference_slots": class_reference_slots,
        "range_type_distribution": dict(sorted(range_types.items(), key=lambda x: x[1], reverse=True)),
        "has_inheritance": bool(inheritance.get("parents") or inheritance.get("is_a")),
        "parent_classes": inheritance.get("parents", []),
        "key_slots": [name for name, slot in slots.items() if slot.get("identifier") or slot.get("key")],
        "most_common_ranges": list(dict(sorted(range_types.items(), key=lambda x: x[1], reverse=True)).keys())[:5]
    }

    return summary


@mcp.tool
def examine_openapi_spec(
        openapi_url: str = "https://api.microbiomedata.org/openapi.json",
        include_schemas: bool = True,
        include_endpoints: bool = True,
        include_examples: bool = True,
        max_endpoints: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Examine and analyze OpenAPI specification to help LLMs understand API structure.

    This function fetches and analyzes an OpenAPI specification (Swagger) JSON file,
    providing structured information about API endpoints, schemas, and capabilities.

    Parameters:
        openapi_url (str): URL to the OpenAPI JSON specification
        include_schemas (bool): Whether to include data model schemas
        include_endpoints (bool): Whether to include endpoint details
        include_examples (bool): Whether to include example values
        max_endpoints (int, optional): Maximum number of endpoints to analyze

    Returns:
        Dict[str, Any]: Comprehensive analysis of the OpenAPI specification
    """
    try:
        # Fetch the OpenAPI specification
        response = requests.get(openapi_url)
        response.raise_for_status()
        spec = response.json()

        analysis = {
            "api_info": extract_api_info(spec),
            "summary": generate_api_summary(spec)
        }

        if include_endpoints:
            analysis["endpoints"] = analyze_endpoints(spec, max_endpoints)

        if include_schemas:
            analysis["schemas"] = analyze_schemas(spec, include_examples)

        analysis["collections"] = extract_collections_info(spec)

        return analysis

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch OpenAPI spec: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in OpenAPI spec: {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing OpenAPI spec: {str(e)}"}


def extract_api_info(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract basic API information from the OpenAPI spec."""
    info = spec.get("info", {})
    return {
        "title": info.get("title", "Unknown API"),
        "version": info.get("version", "Unknown"),
        "description": info.get("description", "No description available"),
        "openapi_version": spec.get("openapi", "Unknown"),
        "servers": spec.get("servers", []),
        "base_paths": [server.get("url", "") for server in spec.get("servers", [])]
    }


def generate_api_summary(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a high-level summary of the API."""
    paths = spec.get("paths", {})
    schemas = spec.get("components", {}).get("schemas", {})

    # Count endpoints by method
    method_counts = {}
    total_endpoints = 0

    for path, methods in paths.items():
        for method in methods.keys():
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                method_counts[method.upper()] = method_counts.get(method.upper(), 0) + 1
                total_endpoints += 1

    # Find collection-like endpoints dynamically
    collection_endpoints = []
    for path in paths.keys():
        path_parts = path.lower().split('/')
        # Look for common collection patterns
        if any(part.endswith('_set') or 'collection' in part for part in path_parts):
            collection_endpoints.append(path)

    return {
        "total_endpoints": total_endpoints,
        "methods_available": list(method_counts.keys()),
        "method_distribution": method_counts,
        "total_schemas": len(schemas),
        "collection_endpoints_count": len(collection_endpoints),
        "appears_to_be_rest_api": "GET" in method_counts
    }


def analyze_endpoints(spec: Dict[str, Any], max_endpoints: Optional[int] = None) -> Dict[str, Any]:
    """Analyze API endpoints in detail."""
    paths = spec.get("paths", {})
    endpoints = []

    count = 0
    for path, methods in paths.items():
        if max_endpoints and count >= max_endpoints:
            break

        for method, details in methods.items():
            if method.lower() in ["get", "post", "put", "delete", "patch"]:
                endpoint_info = {
                    "path": path,
                    "method": method.upper(),
                    "summary": details.get("summary", "No summary"),
                    "description": details.get("description", "No description"),
                    "parameters": analyze_parameters(details.get("parameters", [])),
                    "responses": list(details.get("responses", {}).keys()),
                    "tags": details.get("tags", [])
                }

                # Check if this looks like a collection-like endpoint
                path_parts = path.lower().split('/')
                if any(part.endswith('_set') or 'collection' in part for part in path_parts):
                    endpoint_info["is_collection_endpoint"] = True
                    endpoint_info["collection_name"] = extract_collection_name_from_path(path)

                endpoints.append(endpoint_info)
                count += 1

    # Group endpoints by collection/resource
    collections = {}
    other_endpoints = []

    for endpoint in endpoints:
        if endpoint.get("is_collection_endpoint"):
            collection_name = endpoint.get("collection_name", "unknown")
            if collection_name not in collections:
                collections[collection_name] = []
            collections[collection_name].append(endpoint)
        else:
            other_endpoints.append(endpoint)

    return {
        "total_analyzed": len(endpoints),
        "collection_endpoints": collections,
        "other_endpoints": other_endpoints[:10],  # Limit other endpoints
        "endpoint_patterns": identify_endpoint_patterns(endpoints)
    }


def analyze_parameters(parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze endpoint parameters."""
    if not parameters:
        return {"count": 0, "types": []}

    param_info = {
        "count": len(parameters),
        "by_location": {},
        "required_params": [],
        "optional_params": []
    }

    for param in parameters:
        location = param.get("in", "unknown")
        param_info["by_location"][location] = param_info["by_location"].get(location, 0) + 1

        param_name = param.get("name", "unnamed")
        if param.get("required", False):
            param_info["required_params"].append(param_name)
        else:
            param_info["optional_params"].append(param_name)

    return param_info


def extract_collection_name_from_path(path: str) -> str:
    """Extract collection name from API path using dynamic detection."""
    parts = path.strip("/").split("/")
    for part in parts:
        if "_set" in part:
            return part
        if "collection" in part:
            return part
        # Look for parts that might be collection names (plurals, contain common terms)
        if len(part) > 3 and (part.endswith('s') or len([c for c in part if c.islower()]) > 2):
            return part
    return parts[-1] if parts else "unknown"


def identify_endpoint_patterns(endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify common patterns in the endpoints."""
    patterns = {
        "crud_operations": {},
        "common_path_segments": {},
        "parameter_patterns": {}
    }

    # CRUD patterns
    for endpoint in endpoints:
        method = endpoint["method"]
        path = endpoint["path"]

        if method == "GET" and "{" not in path:
            patterns["crud_operations"]["list"] = patterns["crud_operations"].get("list", 0) + 1
        elif method == "GET" and "{" in path:
            patterns["crud_operations"]["get_by_id"] = patterns["crud_operations"].get("get_by_id", 0) + 1
        elif method == "POST":
            patterns["crud_operations"]["create"] = patterns["crud_operations"].get("create", 0) + 1
        elif method == "PUT":
            patterns["crud_operations"]["update"] = patterns["crud_operations"].get("update", 0) + 1
        elif method == "DELETE":
            patterns["crud_operations"]["delete"] = patterns["crud_operations"].get("delete", 0) + 1

    return patterns


def analyze_schemas(spec: Dict[str, Any], include_examples: bool = True) -> Dict[str, Any]:
    """Analyze data schemas/models in the OpenAPI spec."""
    schemas = spec.get("components", {}).get("schemas", {})

    if not schemas:
        return {"count": 0, "schemas": {}}

    schema_analysis = {
        "count": len(schemas),
        "schemas": {}
    }

    # Dynamically identify important schemas by analyzing their content
    important_schemas = []
    for schema_name, schema_def in schemas.items():
        # Look for schemas with many properties or complex structures
        if (schema_def.get("type") == "object" and
                len(schema_def.get("properties", {})) > 5):
            important_schemas.append(schema_name)
        # Or schemas with meaningful names (not just generic types)
        elif len(schema_name) > 4 and any(c.isupper() for c in schema_name[1:]):
            important_schemas.append(schema_name)

    # Analyze schemas (limit to reasonable number)
    for schema_name in list(schemas.keys())[:20]:
        schema_def = schemas[schema_name]
        schema_info = {
            "type": schema_def.get("type", "unknown"),
            "description": schema_def.get("description", "No description"),
            "is_important": schema_name in important_schemas
        }

        if schema_def.get("type") == "object":
            properties = schema_def.get("properties", {})
            schema_info["property_count"] = len(properties)
            schema_info["required_fields"] = schema_def.get("required", [])

            if include_examples and properties:
                # Show a few key properties
                sample_properties = {}
                for prop_name, prop_def in list(properties.items())[:5]:
                    sample_properties[prop_name] = {
                        "type": prop_def.get("type", "unknown"),
                        "description": prop_def.get("description", "")[:100]
                    }
                schema_info["sample_properties"] = sample_properties

        schema_analysis["schemas"][schema_name] = schema_info

    return schema_analysis


def extract_collections_info(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract information about collections from the API spec using dynamic detection."""
    paths = spec.get("paths", {})
    collections = {}

    for path, methods in paths.items():
        # Look for collection-like paths dynamically
        path_parts = path.lower().split('/')
        if any(part.endswith('_set') or 'collection' in part for part in path_parts):
            collection_name = extract_collection_name_from_path(path)
            if collection_name not in collections:
                collections[collection_name] = {
                    "endpoints": [],
                    "operations": [],
                    "description": ""
                }

            for method, details in methods.items():
                if method.lower() in ["get", "post", "put", "delete", "patch"]:
                    collections[collection_name]["endpoints"].append(f"{method.upper()} {path}")
                    collections[collection_name]["operations"].append(method.upper())

                    # Try to get description from the first endpoint
                    if not collections[collection_name]["description"] and details.get("description"):
                        collections[collection_name]["description"] = details["description"][:200]

    return {
        "discovered_collections": list(collections.keys()),
        "collection_details": collections,
        "total_collections": len(collections)
    }


@mcp.tool
def examine_nmdc_slot_page(
        slot_name: str = "doi_category",
        schema_base_url: str = "https://microbiomedata.github.io/nmdc-schema/",
        verbose: bool = False,
) -> Dict[str, Any]:
    """
    Examine a specific NMDC schema slot page to understand its properties and range.

    This function fetches a slot documentation page and extracts information about
    the slot's range, description, constraints, and valid values.

    Parameters:
        slot_name (str): Name of the slot to examine (e.g., 'doi_category', 'lat_lon')
        schema_base_url (str): Base URL of the NMDC schema documentation
        verbose (bool): If True, print detailed analysis information

    Returns:
        Dict[str, Any]: Analysis of the slot including range, constraints, and valid values
    """
    try:
        # Import BeautifulSoup for HTML parsing
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "BeautifulSoup4 is required. Install with: pip install beautifulsoup4"}

        # Construct the slot URL
        slot_url = f"{schema_base_url.rstrip('/')}/{slot_name}/"

        if verbose:
            print(f"Fetching slot information from: {slot_url}")

        # Fetch the slot documentation page
        response = requests.get(slot_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        analysis = {
            "slot_info": extract_slot_metadata(soup, slot_name),
            "range_info": extract_slot_range_info(soup, verbose),
            "constraints": extract_slot_constraints(soup),
            "usage_info": extract_slot_usage_info(soup),
            "summary": {}
        }

        # Generate summary
        analysis["summary"] = generate_slot_summary(analysis)

        if verbose:
            print(f"Slot analysis complete for: {slot_name}")
            print(f"Range: {analysis['range_info'].get('range', 'Unknown')}")

        return analysis

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch slot documentation for '{slot_name}': {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing slot '{slot_name}': {str(e)}"}


def extract_slot_metadata(soup, slot_name: str) -> Dict[str, Any]:
    """Extract metadata about the slot from its documentation page."""
    metadata = {
        "slot_name": slot_name,
        "description": "No description available",
        "title": "",
        "uri": ""
    }

    # Look for slot title/heading
    title_elem = soup.find('h1') or soup.find('h2')
    if title_elem:
        metadata["title"] = title_elem.get_text().strip()

    # Look for slot description - usually in first paragraph or description section
    desc_candidates = [
        soup.find('p'),
        soup.find('div', class_='description'),
        soup.find('div', id='description'),
        soup.find('section', class_='description')
    ]

    for candidate in desc_candidates:
        if candidate and candidate.get_text().strip():
            metadata["description"] = candidate.get_text().strip()
            break

    # Look for URI/identifier information
    uri_elem = soup.find('code') or soup.find('span', class_='uri')
    if uri_elem:
        metadata["uri"] = uri_elem.get_text().strip()

    return metadata


def extract_slot_range_info(soup, verbose: bool = False) -> Dict[str, Any]:
    """Extract range/type information for the slot."""
    range_info = {
        "range": "Unknown",
        "range_type": "unknown",
        "is_enum": False,
        "enum_values": [],
        "is_class": False,
        "class_name": ""
    }

    # Look for range information in various formats
    # Check for "Range:" label and value
    range_patterns = [
        r'range:\s*([A-Za-z_][A-Za-z0-9_]*)',
        r'type:\s*([A-Za-z_][A-Za-z0-9_]*)',
        r'values?:\s*([A-Za-z_][A-Za-z0-9_]*)'
    ]

    page_text = soup.get_text()
    for pattern in range_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        if matches:
            range_info["range"] = matches[0]
            break

    # Look for range information in structured sections
    # Find any section with "Range" or similar
    range_sections = soup.find_all(['div', 'section', 'table'],
                                   string=lambda text: text and 'range' in text.lower())

    for section in range_sections:
        # Look for links that might indicate the range type
        links = section.find_all('a')
        for link in links:
            link_text = link.get_text().strip()
            href = link.get('href', '')

            # Check if this points to an enum
            if 'enum' in href.lower() or link_text.endswith('Enum'):
                range_info["range"] = link_text
                range_info["is_enum"] = True
                range_info["range_type"] = "enum"
                if verbose:
                    print(f"Found enum range: {link_text}")
            # Check if this points to a class
            elif href and not href.startswith('http') and len(link_text) > 2:
                range_info["range"] = link_text
                range_info["is_class"] = True
                range_info["class_name"] = link_text
                range_info["range_type"] = "class"
                if verbose:
                    print(f"Found class range: {link_text}")

    # Look in tables for structured range information
    tables = soup.find_all('table')
    for table in tables:
        range_data = extract_range_from_table(table, verbose)
        if range_data:
            range_info.update(range_data)

    return range_info


def extract_range_from_table(table, verbose: bool = False) -> Dict[str, Any]:
    """Extract range information from a table."""
    range_data = {}

    rows = table.find_all('tr')
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 2:
            key = cells[0].get_text().strip().lower()
            value_cell = cells[1]

            if 'range' in key:
                # Look for links in the value cell
                link = value_cell.find('a')
                if link:
                    range_value = link.get_text().strip()
                    href = link.get('href', '')

                    range_data["range"] = range_value
                    if 'enum' in href.lower():
                        range_data["is_enum"] = True
                        range_data["range_type"] = "enum"
                    else:
                        range_data["is_class"] = True
                        range_data["class_name"] = range_value
                        range_data["range_type"] = "class"
                else:
                    range_data["range"] = value_cell.get_text().strip()

                if verbose:
                    print(f"Found range in table: {range_data['range']}")
                break

    return range_data


def extract_slot_constraints(soup) -> Dict[str, Any]:
    """Extract constraints and validation rules for the slot."""
    constraints = {
        "required": False,
        "multivalued": False,
        "pattern": "",
        "minimum_value": None,
        "maximum_value": None,
        "valid_values": []
    }

    # Look for constraint information in tables or structured sections
    tables = soup.find_all('table')

    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text().strip().lower()
                value = cells[1].get_text().strip()

                if 'required' in key:
                    constraints["required"] = value.lower() in ['true', 'yes', '1', 'required']
                elif 'multivalued' in key or 'multi' in key:
                    constraints["multivalued"] = value.lower() in ['true', 'yes', '1']
                elif 'pattern' in key:
                    constraints["pattern"] = value
                elif 'minimum' in key or 'min' in key:
                    try:
                        constraints["minimum_value"] = float(value)
                    except ValueError:
                        pass
                elif 'maximum' in key or 'max' in key:
                    try:
                        constraints["maximum_value"] = float(value)
                    except ValueError:
                        pass

    return constraints


def extract_slot_usage_info(soup) -> Dict[str, Any]:
    """Extract information about how the slot is used."""
    usage = {
        "used_by_classes": [],
        "examples": [],
        "notes": ""
    }

    # Look for "Used by" sections
    page_text = soup.get_text()
    if 'used by' in page_text.lower():
        # Try to find classes that use this slot
        used_by_section = soup.find(string=lambda text: text and 'used by' in text.lower())
        if used_by_section:
            parent = used_by_section.parent
            if parent:
                links = parent.find_all('a')
                for link in links:
                    class_name = link.get_text().strip()
                    if class_name and len(class_name) > 2 and class_name[0].isupper():
                        usage["used_by_classes"].append(class_name)

    # Look for examples
    example_indicators = ['example', 'sample', 'instance']
    for indicator in example_indicators:
        if indicator in page_text.lower():
            # Try to extract example values
            example_section = soup.find(string=lambda text: text and indicator in text.lower())
            if example_section:
                parent = example_section.parent
                if parent:
                    # Look for code blocks or formatted text
                    code_blocks = parent.find_all(['code', 'pre'])
                    for block in code_blocks:
                        example_text = block.get_text().strip()
                        if example_text and len(example_text) < 100:
                            usage["examples"].append(example_text)

    return usage


def generate_slot_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the slot analysis."""
    slot_info = analysis.get("slot_info", {})
    range_info = analysis.get("range_info", {})
    constraints = analysis.get("constraints", {})
    usage = analysis.get("usage_info", {})

    summary = {
        "slot_name": slot_info.get("slot_name", "Unknown"),
        "range": range_info.get("range", "Unknown"),
        "range_type": range_info.get("range_type", "unknown"),
        "is_enum": range_info.get("is_enum", False),
        "is_required": constraints.get("required", False),
        "is_multivalued": constraints.get("multivalued", False),
        "used_by_class_count": len(usage.get("used_by_classes", [])),
        "has_examples": len(usage.get("examples", [])) > 0,
        "has_constraints": any([
            constraints.get("pattern"),
            constraints.get("minimum_value") is not None,
            constraints.get("maximum_value") is not None
        ]),
        "analysis_completeness": "partial"
    }

    # Determine analysis completeness
    if (summary["range"] != "Unknown" and
            summary["used_by_class_count"] > 0):
        if summary["has_examples"] or summary["has_constraints"]:
            summary["analysis_completeness"] = "comprehensive"
        else:
            summary["analysis_completeness"] = "good"

    return summary


@mcp.tool
def examine_nmdc_enum_page(
        enum_name: str = "DoiCategoryEnum",
        schema_base_url: str = "https://microbiomedata.github.io/nmdc-schema/",
        verbose: bool = False,
) -> Dict[str, Any]:
    """
    Examine a specific NMDC schema enum page to understand its valid values.

    This function fetches an enum documentation page and extracts information about
    the enum's valid values, descriptions, and usage.

    Parameters:
        enum_name (str): Name of the enum to examine (e.g., 'DoiCategoryEnum', 'EcosystemEnum')
        schema_base_url (str): Base URL of the NMDC schema documentation
        verbose (bool): If True, print detailed analysis information

    Returns:
        Dict[str, Any]: Analysis of the enum including valid values and descriptions
    """
    try:
        # Import BeautifulSoup for HTML parsing
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "BeautifulSoup4 is required. Install with: pip install beautifulsoup4"}

        # Construct the enum URL
        enum_url = f"{schema_base_url.rstrip('/')}/{enum_name}/"

        if verbose:
            print(f"Fetching enum information from: {enum_url}")

        # Fetch the enum documentation page
        response = requests.get(enum_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        analysis = {
            "enum_info": extract_enum_metadata(soup, enum_name),
            "valid_values": extract_enum_values(soup, verbose),
            "value_descriptions": extract_enum_value_descriptions(soup, verbose),
            "usage_info": extract_enum_usage_info(soup),
            "summary": {}
        }

        # Generate summary
        analysis["summary"] = generate_enum_summary(analysis)

        if verbose:
            value_count = len(analysis["valid_values"])
            print(f"Enum analysis complete for: {enum_name}")
            print(f"Found {value_count} valid values")

        return analysis

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch enum documentation for '{enum_name}': {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing enum '{enum_name}': {str(e)}"}


def extract_enum_metadata(soup, enum_name: str) -> Dict[str, Any]:
    """Extract metadata about the enum from its documentation page."""
    metadata = {
        "enum_name": enum_name,
        "description": "No description available",
        "title": "",
        "uri": ""
    }

    # Look for enum title/heading
    title_elem = soup.find('h1') or soup.find('h2')
    if title_elem:
        metadata["title"] = title_elem.get_text().strip()

    # Look for enum description
    desc_candidates = [
        soup.find('p'),
        soup.find('div', class_='description'),
        soup.find('div', id='description'),
        soup.find('section', class_='description')
    ]

    for candidate in desc_candidates:
        if candidate and candidate.get_text().strip():
            metadata["description"] = candidate.get_text().strip()
            break

    # Look for URI/identifier information
    uri_elem = soup.find('code') or soup.find('span', class_='uri')
    if uri_elem:
        metadata["uri"] = uri_elem.get_text().strip()

    return metadata


def extract_enum_values(soup, verbose: bool = False) -> List[str]:
    """Extract the valid values from the enum page."""
    values = []

    # Look for values in various formats
    # 1. Look for tables with permissible values
    tables = soup.find_all('table')
    for table in tables:
        table_values = extract_values_from_table(table, verbose)
        if table_values:
            values.extend(table_values)

    # 2. Look for lists of values
    lists = soup.find_all(['ul', 'ol'])
    for lst in lists:
        list_values = extract_values_from_list(lst, verbose)
        if list_values:
            values.extend(list_values)

    # 3. Look for values in text using patterns
    if not values:
        text_values = extract_values_from_text(soup, verbose)
        values.extend(text_values)

    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)

    if verbose:
        print(f"Extracted {len(unique_values)} unique enum values")

    return unique_values


def extract_values_from_table(table, verbose: bool = False) -> List[str]:
    """Extract enum values from a table."""
    values = []

    # Look for table headers to identify the value column
    headers = table.find_all('th')
    header_texts = [th.get_text().strip().lower() for th in headers]

    value_col_index = -1
    for i, header in enumerate(header_texts):
        if any(keyword in header for keyword in ['value', 'permissible', 'allowed', 'valid', 'text']):
            value_col_index = i
            break

    # If no specific value column found, assume first column
    if value_col_index == -1:
        value_col_index = 0

    rows = table.find_all('tr')
    for row in rows[1:]:  # Skip header row
        cells = row.find_all(['td', 'th'])
        if len(cells) > value_col_index:
            cell_text = cells[value_col_index].get_text().strip()
            # Clean up the value
            if cell_text and not cell_text.lower() in ['value', 'description', 'meaning']:
                # Remove quotes and extra whitespace
                clean_value = cell_text.strip('"\'').strip()
                if clean_value and len(clean_value) > 0:
                    values.append(clean_value)
                    if verbose:
                        print(f"Found table value: {clean_value}")

    return values


def extract_values_from_list(lst, verbose: bool = False) -> List[str]:
    """Extract enum values from a list element."""
    values = []

    items = lst.find_all('li')
    for item in items:
        text = item.get_text().strip()

        # Look for patterns like "value: description" or just "value"
        if ':' in text:
            value_part = text.split(':')[0].strip()
        else:
            value_part = text.strip()

        # Clean up the value
        clean_value = value_part.strip('"\'').strip()

        # Validate that this looks like an enum value
        if (clean_value and
                len(clean_value) > 0 and
                len(clean_value) < 100 and  # Reasonable length
                not clean_value.lower().startswith(('the ', 'this ', 'a ', 'an '))):  # Not a description
            values.append(clean_value)
            if verbose:
                print(f"Found list value: {clean_value}")

    return values


def extract_values_from_text(soup, verbose: bool = False) -> List[str]:
    """Extract enum values from free text using patterns."""
    values = []

    text_content = soup.get_text()

    # Look for common enum value patterns
    patterns = [
        r'permissible values?:\s*([^.]+)',
        r'allowed values?:\s*([^.]+)',
        r'valid values?:\s*([^.]+)',
        r'possible values?:\s*([^.]+)',
        r'values?:\s*([^.]+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for match in matches:
            # Split by common separators
            potential_values = re.split(r'[,;|]', match)
            for potential_value in potential_values:
                clean_value = potential_value.strip().strip('"\'').strip()
                if (clean_value and
                        len(clean_value) > 0 and
                        len(clean_value) < 50 and
                        not any(char in clean_value for char in ['\n', '\t'])):
                    values.append(clean_value)
                    if verbose:
                        print(f"Found text pattern value: {clean_value}")

    return values


def extract_enum_value_descriptions(soup, verbose: bool = False) -> Dict[str, str]:
    """Extract descriptions for enum values if available."""
    descriptions = {}

    # Look for tables that have both values and descriptions
    tables = soup.find_all('table')
    for table in tables:
        table_descriptions = extract_descriptions_from_table(table, verbose)
        descriptions.update(table_descriptions)

    # Look for definition lists
    dl_elements = soup.find_all('dl')
    for dl in dl_elements:
        dl_descriptions = extract_descriptions_from_definition_list(dl, verbose)
        descriptions.update(dl_descriptions)

    return descriptions


def extract_descriptions_from_table(table, verbose: bool = False) -> Dict[str, str]:
    """Extract value-description pairs from a table."""
    descriptions = {}

    # Get headers to identify columns
    headers = table.find_all('th')
    header_texts = [th.get_text().strip().lower() for th in headers]

    value_col = -1
    desc_col = -1

    for i, header in enumerate(header_texts):
        if any(keyword in header for keyword in ['value', 'permissible', 'text']):
            value_col = i
        elif any(keyword in header for keyword in ['description', 'meaning', 'definition', 'desc']):
            desc_col = i

    if value_col >= 0 and desc_col >= 0:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header
            cells = row.find_all(['td', 'th'])
            if len(cells) > max(value_col, desc_col):
                value = cells[value_col].get_text().strip().strip('"\'')
                desc = cells[desc_col].get_text().strip()

                if value and desc:
                    descriptions[value] = desc
                    if verbose:
                        print(f"Found description for {value}: {desc[:50]}...")

    return descriptions


def extract_descriptions_from_definition_list(dl, verbose: bool = False) -> Dict[str, str]:
    """Extract value-description pairs from a definition list."""
    descriptions = {}

    dt_elements = dl.find_all('dt')
    for dt in dt_elements:
        value = dt.get_text().strip().strip('"\'')
        dd = dt.find_next_sibling('dd')
        if dd:
            desc = dd.get_text().strip()
            descriptions[value] = desc
            if verbose:
                print(f"Found definition for {value}: {desc[:50]}...")

    return descriptions


def extract_enum_usage_info(soup) -> Dict[str, Any]:
    """Extract information about how the enum is used."""
    usage = {
        "used_by_slots": [],
        "used_by_classes": [],
        "notes": ""
    }

    # Look for "Used by" sections
    page_text = soup.get_text()
    if 'used by' in page_text.lower():
        used_by_section = soup.find(string=lambda text: text and 'used by' in text.lower())
        if used_by_section:
            parent = used_by_section.parent
            if parent:
                links = parent.find_all('a')
                for link in links:
                    link_text = link.get_text().strip()
                    href = link.get('href', '')

                    if '/slots/' in href:
                        usage["used_by_slots"].append(link_text)
                    elif link_text and len(link_text) > 2 and link_text[0].isupper():
                        usage["used_by_classes"].append(link_text)

    return usage


def generate_enum_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the enum analysis."""
    enum_info = analysis.get("enum_info", {})
    valid_values = analysis.get("valid_values", [])
    descriptions = analysis.get("value_descriptions", {})
    usage = analysis.get("usage_info", {})

    summary = {
        "enum_name": enum_info.get("enum_name", "Unknown"),
        "total_values": len(valid_values),
        "values_with_descriptions": len(descriptions),
        "has_complete_descriptions": len(descriptions) == len(valid_values) and len(valid_values) > 0,
        "used_by_slots_count": len(usage.get("used_by_slots", [])),
        "used_by_classes_count": len(usage.get("used_by_classes", [])),
        "sample_values": valid_values[:5] if valid_values else [],
        "all_values": valid_values,
        "analysis_quality": "unknown"
    }

    # Determine analysis quality
    if summary["total_values"] > 0:
        if summary["has_complete_descriptions"] and summary["used_by_slots_count"] > 0:
            summary["analysis_quality"] = "excellent"
        elif summary["values_with_descriptions"] > 0 or summary["used_by_slots_count"] > 0:
            summary["analysis_quality"] = "good"
        else:
            summary["analysis_quality"] = "basic"
    else:
        summary["analysis_quality"] = "poor"

    return summary


if __name__ == "__main__":
    mcp.run()