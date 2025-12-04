import re
import logging
from typing import List, Dict, Any, Optional

# Imports from other project modules
from config import (
    SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET,
    FEDEX_CLIENT_ID, FEDEX_CLIENT_SECRET
)
from api_clients import ShipStationAPI, FedExAPI, get_order_status_from_pompeii3
from entity_extractor import ExtractedEntity


def create_unified_order_data(
    pompeii_data: Optional[Dict[str, Any]],
    shipstation_data: Optional[Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """
    Creates unified order data from multiple sources (Pompeii3 and ShipStation).
    """
    log = logger or logging.getLogger(__name__)

    # Ensure data sources are dictionaries if they exist, or treat as None
    pompeii_data = pompeii_data if isinstance(pompeii_data, dict) else None
    shipstation_data = shipstation_data if isinstance(shipstation_data, dict) else None

    has_pompeii = pompeii_data and not pompeii_data.get('error')
    has_shipstation = shipstation_data is not None

    if not has_pompeii and not has_shipstation:
        log.warning("No valid data from Pompeii3 or ShipStation to create unified order data.")
        return None

    unified: Dict[str, Any] = {}
    data_sources: List[str] = []

    # Helper to add data if it exists and is not None or empty string
    def add_if_valid(target_dict: Dict[str, Any], key: str, source_data: Optional[Dict[str, Any]], source_key: str):
        if source_data and source_key in source_data:
            value = source_data[source_key]
            if value is not None and value != '':
                target_dict[key] = value

    # --- Populate Unified Data ---
    # Priority: ShipStation > Pompeii3 for common fields
    
    # Order ID and Number
    if has_shipstation:
        add_if_valid(unified, 'order_id', shipstation_data, 'order_id')
        add_if_valid(unified, 'order_number', shipstation_data, 'order_number')
    elif has_pompeii:
        add_if_valid(unified, 'order_id', pompeii_data, 'order_id_internal')
        add_if_valid(unified, 'order_number', pompeii_data, 'order_number')

    # Order Date
    if has_shipstation:
        add_if_valid(unified, 'order_date', shipstation_data, 'order_date')

    # Recipient Name
    if has_shipstation:
        add_if_valid(unified, 'recipient_name', shipstation_data, 'recipient_name')
    elif has_pompeii:
        add_if_valid(unified, 'recipient_name', pompeii_data, 'recipient_name')

    # Order Total
    if has_shipstation:
        add_if_valid(unified, 'order_total', shipstation_data, 'order_total')

    # Order Status
    if has_shipstation:
        add_if_valid(unified, 'order_status', shipstation_data, 'order_status')
    elif has_pompeii:
        add_if_valid(unified, 'order_status', pompeii_data, 'order_status')

    # Ship By Date / Estimated Ship Date
    if has_shipstation:
        add_if_valid(unified, 'ship_by_date', shipstation_data, 'ship_by_date')
    if has_pompeii:
        add_if_valid(unified, 'estimated_ship_date', pompeii_data, 'estimated_ship_date')

    # Shipping Details (Address) - Primarily from ShipStation
    if has_shipstation and shipstation_data.get('shipping_details'):
        unified['shipping_details'] = shipstation_data['shipping_details']

    # Pompeii3 specific fields
    if has_pompeii:
        data_sources.append('pompeii3_api')
        add_if_valid(unified, 'shipping_method', pompeii_data, 'shipping_method')
        add_if_valid(unified, 'shipping_class', pompeii_data, 'shipping_class')
        add_if_valid(unified, 'production_percentage', pompeii_data, 'production_percentage')
        add_if_valid(unified, 'pompeii3_tracking_link', pompeii_data, 'tracking_link')

    # Items - Attempt to merge or prioritize ShipStation if richer
    unified_items = []
    shipstation_item_skus = set()
    
    if has_shipstation and shipstation_data.get('items'):
        data_sources.append('shipstation_api')
        for item in shipstation_data['items']:
            if item and item.get('item_sku'):
                shipstation_item_skus.add(item['item_sku'])
            processed_item = {
                'item_name': item.get('item_name'),
                'item_sku': item.get('item_sku'),
                'quantity': item.get('quantity'),
                'unit_price': item.get('unit_price'),
                'image_url': item.get('image_url'),
                'source': 'shipstation_api'
            }
            unified_items.append({k: v for k, v in processed_item.items() if v is not None})

    if has_pompeii and pompeii_data.get('items'):
        for item in pompeii_data['items']:
            if item and (not item.get('sku') or item.get('sku') not in shipstation_item_skus):
                processed_item = {
                    'item_name': item.get('name'),
                    'item_sku': item.get('sku'),
                    'quantity': item.get('quantity'),
                    'source': 'pompeii3_api'
                }
                unified_items.append({k: v for k, v in processed_item.items() if v is not None})
    
    unified['items'] = unified_items

    # Collect all unique tracking numbers from all sources
    all_tracking_numbers_info = []
    seen_tracking_numbers = set()

    if has_pompeii and pompeii_data.get('tracking_number'):
        tn = pompeii_data['tracking_number']
        if tn not in seen_tracking_numbers:
            all_tracking_numbers_info.append({
                'tracking_number': tn,
                'carrier': pompeii_data.get('shipping_method'),
                'service': pompeii_data.get('shipping_class'),
                'source': 'pompeii3_api'
            })
            seen_tracking_numbers.add(tn)

    if has_shipstation and shipstation_data.get('tracking'):
        for track_info in shipstation_data['tracking']:
            tn = track_info.get('tracking_number')
            if tn and tn not in seen_tracking_numbers:
                all_tracking_numbers_info.append({
                    'tracking_number': tn,
                    'carrier': track_info.get('carrier'),
                    'service': track_info.get('service'),
                    'ship_date': track_info.get('ship_date'),
                    'source': 'shipstation_api'
                })
                seen_tracking_numbers.add(tn)
    
    if all_tracking_numbers_info:
        unified['tracking_info_list'] = all_tracking_numbers_info

    unified['data_sources_used'] = list(set(data_sources))
    log.info(f"Unified order data created from sources: {unified['data_sources_used']}. Order: {unified.get('order_number')}")
    return unified


def get_order_info(
    order_entities: List[ExtractedEntity],
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Fetches and combines order information from Pompeii3 and ShipStation for given order number entities.
    Returns a list of processed order data, where each item corresponds to an order.
    """
    log = logger or logging.getLogger(__name__)
    processed_orders_data: List[Dict[str, Any]] = []

    if not order_entities:
        log.warning("No order entities provided to get_order_info.")
        return processed_orders_data

    order_numbers_to_query = []
    for entity in order_entities:
        if entity.type == "ORDER_NUMBER" and entity.value:
            clean_order_number = re.sub(r'[^\w-]', '', str(entity.value))
            if clean_order_number:
                order_numbers_to_query.append(clean_order_number)

    if not order_numbers_to_query:
        log.warning("No valid order numbers found in entities after cleaning.")
        return processed_orders_data
    
    unique_order_numbers = sorted(list(set(order_numbers_to_query)))
    log.info(f"Querying order information for order numbers: {unique_order_numbers}")

    try:
        shipstation_client = ShipStationAPI(SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET, logger=log)
    except Exception as e:
        log.error(f"Failed to initialize ShipStation client: {e}")
        shipstation_client = None

    for order_number_str in unique_order_numbers:
        log.info(f"Processing order number: {order_number_str}")
        pompeii_data: Optional[Dict[str, Any]] = None
        shipstation_raw_data: Optional[Dict[str, Any]] = None
        shipstation_processed_data: Optional[Dict[str, Any]] = None

        digits = len(re.sub(r'\D', '', order_number_str))
        
        call_pompeii = digits in [4, 6, 8, 15] or not order_number_str.isdigit()
        call_shipstation = (digits == 8 and order_number_str.isdigit()) or (6 <= digits <= 10 and order_number_str.isdigit())

        log.info(f"Order number '{order_number_str}' (digits: {digits}): Call Pompeii3={call_pompeii}, Call ShipStation={call_shipstation}")

        if call_pompeii:
            log.info(f"Calling Pompeii3 API for order/PO: {order_number_str}")
            try:
                pompeii_data = get_order_status_from_pompeii3(order_number_str, log)
                if pompeii_data and not pompeii_data.get('error'):
                    log.info(f"Successfully retrieved data from Pompeii3 for {order_number_str}.")
                else:
                    log.warning(f"Failed to get valid data from Pompeii3 for {order_number_str}. Error: {pompeii_data.get('error') if pompeii_data else 'No data'}")
            except Exception as e:
                log.error(f"Error calling Pompeii3 API for {order_number_str}: {e}")
                pompeii_data = None

        if call_shipstation and shipstation_client:
            log.info(f"Calling ShipStation API for order: {order_number_str}")
            try:
                shipstation_raw_data = shipstation_client.get_order_by_order_number(order_number_str)
                if shipstation_raw_data:
                    shipstation_processed_data = shipstation_client.process_order_data(shipstation_raw_data)
                    log.info(f"Successfully retrieved and processed data from ShipStation for {order_number_str}.")
                    
                    # Fetch tracking numbers from ShipStation separately using the orderId
                    ss_order_id = shipstation_processed_data.get('order_id')
                    if ss_order_id:
                        tracking_list = shipstation_client.get_tracking_numbers(ss_order_id)
                        shipstation_processed_data['tracking'] = tracking_list
                else:
                    log.warning(f"No data found in ShipStation for order {order_number_str}.")
            except Exception as e:
                log.error(f"Error calling ShipStation API for {order_number_str}: {e}")
                shipstation_processed_data = None
        
        # Create unified data for this order
        current_order_unified_data = create_unified_order_data(pompeii_data, shipstation_processed_data, log)

        if current_order_unified_data:
            final_order_object = {
                'original_query_order_number': order_number_str,
                'unified_data': current_order_unified_data,
                'pompeii3_source_data': pompeii_data if (pompeii_data and not pompeii_data.get('error')) else None,
                'shipstation_source_data': shipstation_processed_data
            }
            
            # Copy key fields from unified_data to top level for easier access
            for key in ['order_id', 'order_number', 'order_date', 'recipient_name',
                        'order_total', 'order_status', 'ship_by_date', 'shipping_details', 'items', 'tracking_info_list']:
                if key in current_order_unified_data:
                    final_order_object[key] = current_order_unified_data[key]
            
            processed_orders_data.append(final_order_object)
        else:
            log.warning(f"Could not create unified data for order number {order_number_str}. No valid data from any source.")
            processed_orders_data.append({
                'original_query_order_number': order_number_str,
                'error': 'No order information found from any API.'
            })

    log.info(f"Finished processing. Returning data for {len(processed_orders_data)} queried order numbers.")
    return processed_orders_data


def get_purchase_order_info(
    po_entities: List[ExtractedEntity],
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Fetches order information specifically for Purchase Order numbers using the Pompeii3 API.
    """
    log = logger or logging.getLogger(__name__)
    processed_po_data: List[Dict[str, Any]] = []

    if not po_entities:
        log.warning("No Purchase Order entities provided.")
        return processed_po_data

    po_numbers_to_query = []
    for entity in po_entities:
        if entity.type == "PURCHASE_ORDER_NUMBER" and entity.value:
            clean_po_number = re.sub(r'\D', '', str(entity.value))
            if len(clean_po_number) == 15:
                po_numbers_to_query.append(clean_po_number)
            else:
                log.warning(f"Skipping invalid PO entity value: '{entity.value}' (cleaned: '{clean_po_number}'). Not 15 digits.")

    if not po_numbers_to_query:
        log.warning("No valid 15-digit Purchase Order numbers found in entities.")
        return processed_po_data

    unique_po_numbers = sorted(list(set(po_numbers_to_query)))
    log.info(f"Querying Pompeii3 for Purchase Order numbers: {unique_po_numbers}")

    for po_number_str in unique_po_numbers:
        log.info(f"Processing Purchase Order: {po_number_str}")
        try:
            pompeii_po_data = get_order_status_from_pompeii3(po_number_str, log)

            if pompeii_po_data and not pompeii_po_data.get('error'):
                log.info(f"Successfully retrieved data from Pompeii3 for PO {po_number_str}.")
                final_po_object = {
                    'original_query_po_number': po_number_str,
                    'is_purchase_order': True,
                    'unified_data': {
                        'order_number': po_number_str,
                        'purchase_order_number': po_number_str,
                        'order_status': pompeii_po_data.get('order_status'),
                        'recipient_name': pompeii_po_data.get('recipient_name'),
                        'shipping_method': pompeii_po_data.get('shipping_method'),
                        'shipping_class': pompeii_po_data.get('shipping_class'),
                        'estimated_ship_date': pompeii_po_data.get('estimated_ship_date'),
                        'tracking_number': pompeii_po_data.get('tracking_number'),
                        'tracking_link': pompeii_po_data.get('tracking_link'),
                        'items': pompeii_po_data.get('items', []),
                        'data_sources_used': ['pompeii3_api_po_lookup']
                    },
                    'pompeii3_source_data': pompeii_po_data
                }
                
                # Copy key fields to top level
                for key in ['order_status', 'recipient_name', 'items', 'tracking_number', 'tracking_link']:
                    if key in final_po_object['unified_data']:
                        final_po_object[key] = final_po_object['unified_data'][key]

                processed_po_data.append(final_po_object)
            else:
                log.warning(f"Failed to get valid data from Pompeii3 for PO {po_number_str}. Error: {pompeii_po_data.get('error') if pompeii_po_data else 'No data'}")
                processed_po_data.append({
                    'original_query_po_number': po_number_str,
                    'is_purchase_order': True,
                    'error': f"No PO information found from Pompeii3 API for {po_number_str}."
                })
        except Exception as e:
            log.error(f"Error processing PO {po_number_str}: {e}")
            processed_po_data.append({
                'original_query_po_number': po_number_str,
                'is_purchase_order': True,
                'error': f"Error processing PO {po_number_str}: {str(e)}"
            })
            
    log.info(f"Finished PO processing. Returning data for {len(processed_po_data)} queried POs.")
    return processed_po_data


def get_tracking_info(
    tracking_entities: List[ExtractedEntity],
    logger: Optional[logging.Logger] = None
) -> List[Optional[Dict[str, Any]]]:
    """
    Fetches detailed tracking information for given tracking number entities.
    """
    log = logger or logging.getLogger(__name__)
    detailed_tracking_results: List[Optional[Dict[str, Any]]] = []

    if not tracking_entities:
        log.warning("No tracking entities provided.")
        return detailed_tracking_results

    tracking_numbers_to_query = []
    for entity in tracking_entities:
        if entity.type == "TRACKING_NUMBER" and entity.value:
            tracking_numbers_to_query.append(entity.value)
    
    unique_tracking_numbers = sorted(list(set(tracking_numbers_to_query)))
    if not unique_tracking_numbers:
        log.warning("No valid tracking numbers found in entities.")
        return detailed_tracking_results

    log.info(f"Querying tracking information for: {unique_tracking_numbers}")
    
    try:
        fedex_client = FedExAPI(FEDEX_CLIENT_ID, FEDEX_CLIENT_SECRET, sandbox=True, logger=log)
    except Exception as e:
        log.error(f"Failed to initialize FedEx client: {e}")
        return detailed_tracking_results

    for tn_str in unique_tracking_numbers:
        log.info(f"Calling FedEx API for tracking number: {tn_str}")
        try:
            raw_tracking_data = fedex_client.get_tracking_info([tn_str])
            if raw_tracking_data:
                processed_tracking_data = fedex_client.process_tracking_data(raw_tracking_data)
                if processed_tracking_data:
                    log.info(f"Successfully retrieved and processed tracking for {tn_str}.")
                    detailed_tracking_results.append(processed_tracking_data)
                else:
                    log.warning(f"Failed to process tracking data from FedEx for {tn_str}, though API call was successful.")
                    detailed_tracking_results.append({'tracking_number': tn_str, 'error': 'Failed to process FedEx response'})
            else:
                log.warning(f"No tracking data returned from FedEx API for {tn_str}.")
                detailed_tracking_results.append({'tracking_number': tn_str, 'error': 'No data from FedEx API'})
        except Exception as e:
            log.error(f"Error getting/processing FedEx tracking info for {tn_str}: {e}")
            detailed_tracking_results.append({'tracking_number': tn_str, 'error': str(e)})
            
    log.info(f"Returning {len(detailed_tracking_results)} tracking results.")
    return detailed_tracking_results



def process_order_tracking_info(email_data: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Enhanced order processing that integrates with ShipStation API.
    Tries unified search first, then order number specific search, then falls back to customer name search.
    """
    log = logger or logging.getLogger(__name__)
    
    # Initialize ShipStation API client
    try:
        from config import SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET
        
        if not SHIPSTATION_API_KEY or not SHIPSTATION_API_SECRET:
            log.warning("ShipStation API credentials not available. Skipping order processing.")
            return
            
        shipstation_api = ShipStationAPI(SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET, logger=log)
        
    except ImportError as e:
        log.error(f"Failed to import ShipStation API: {e}")
        return
    except Exception as e:
        log.error(f"Failed to initialize ShipStation API: {e}")
        return
    
    # Extract relevant information from email_data
    entities = email_data.get('entities', [])
    sender_name = email_data.get('sender_name', '')
    
    # Extract order numbers and customer names from entities
    order_numbers = []
    customer_names = []
    
    for entity in entities:
        if entity.get('type') in ['ORDER_NUMBER', 'PURCHASE_ORDER_NUMBER']:
            order_numbers.append(entity.get('value'))
        elif entity.get('type') == 'CUSTOMER_NAME':
            customer_names.append(entity.get('value'))
    
    # Add sender name as fallback customer name if not in entities
    if sender_name and sender_name not in customer_names:
        customer_names.append(sender_name)
    
    log.info(f"Processing order info - Order Numbers: {order_numbers}, Customer Names: {customer_names}")
    
    # Initialize order_info and tracking_info lists
    email_data['order_info'] = []
    email_data['tracking_info'] = []
    
    # Strategy 0: Unified search using get_order_by_customer_or_number
    orders_found = False
    all_identifiers = order_numbers + customer_names
    unique_identifiers = list(set([str(id).strip() for id in all_identifiers if id and str(id).strip()]))
    
    for identifier in unique_identifiers:
        if not identifier or len(identifier.strip()) < 2:
            continue
            
        log.info(f"Trying unified search for identifier: {identifier}")
        try:
            # Use the existing get_order_by_customer_or_number method
            order_data = shipstation_api.get_order_by_customer_or_number(identifier)
            if order_data:
                log.info(f"Found order via unified search for: {identifier}")
                
                # Process the order data using existing logic
                processed_order = shipstation_api.process_order_data(order_data)
                
                if processed_order:
                    # Add unified data structure
                    unified_order_data = {
                        'source': 'shipstation',
                        'search_method': 'unified_search',
                        'search_value': identifier,
                        'order_id': processed_order.get('order_id'),
                        'order_number': processed_order.get('order_number'),
                        'order_status': processed_order.get('order_status'),
                        'order_date': processed_order.get('order_date'),
                        'order_total': processed_order.get('order_total'),
                        'recipient_name': processed_order.get('recipient_name'),
                        'customer_email': processed_order.get('customer_email'),
                        'shipping_details': processed_order.get('shipping_details', {}),
                        'items': processed_order.get('items', []),
                        'ship_by_date': processed_order.get('ship_by_date'),
                        'unified_data': processed_order
                    }
                    
                    email_data['order_info'].append(unified_order_data)
                    orders_found = True
                    
                    # Get tracking information for this order
                    if processed_order.get('order_id'):
                        tracking_numbers = shipstation_api.get_tracking_numbers(processed_order['order_id'])
                        
                        if tracking_numbers:
                            log.info(f"Found {len(tracking_numbers)} tracking numbers for identifier {identifier}")
                            
                            for tracking_data in tracking_numbers:
                                tracking_info = {
                                    'source': 'shipstation',
                                    'search_method': 'unified_search',
                                    'tracking_number': tracking_data.get('tracking_number'),
                                    'carrier': tracking_data.get('carrier'),
                                    'service': tracking_data.get('service'),
                                    'ship_date': tracking_data.get('ship_date'),
                                    'order_id': processed_order.get('order_id'),
                                    'order_number': processed_order.get('order_number')
                                }
                                email_data['tracking_info'].append(tracking_info)
                        else:
                            log.info(f"No tracking numbers found for identifier {identifier}")
                            
        except Exception as e:
            log.error(f"Error in unified search for {identifier}: {e}")
            continue
    
    # Strategy 1: Try to find orders by order number first (if unified search didn't find anything)
    if not orders_found:
        for order_number in order_numbers:
            if not order_number:
                continue
                
            log.info(f"Searching for order by number: {order_number}")
            
            try:
                # Use the existing get_order_by_order_number method
                order_data = shipstation_api.get_order_by_order_number(order_number)
                
                if order_data:
                    log.info(f"Found order by number: {order_number}")
                    
                    # Process the order data
                    processed_order = shipstation_api.process_order_data(order_data)
                    
                    if processed_order:
                        # Add unified data structure
                        unified_order_data = {
                            'source': 'shipstation',
                            'order_id': processed_order.get('order_id'),
                            'order_number': processed_order.get('order_number'),
                            'order_status': processed_order.get('order_status'),
                            'order_date': processed_order.get('order_date'),
                            'order_total': processed_order.get('order_total'),
                            'recipient_name': processed_order.get('recipient_name'),
                            'customer_email': processed_order.get('customer_email'),
                            'shipping_details': processed_order.get('shipping_details', {}),
                            'items': processed_order.get('items', []),
                            'ship_by_date': processed_order.get('ship_by_date'),
                            'unified_data': processed_order
                        }
                        
                        email_data['order_info'].append(unified_order_data)
                        orders_found = True
                        
                        # Get tracking information for this order
                        if processed_order.get('order_id'):
                            tracking_numbers = shipstation_api.get_tracking_numbers(processed_order['order_id'])
                            
                            if tracking_numbers:
                                log.info(f"Found {len(tracking_numbers)} tracking numbers for order {order_number}")
                                
                                for tracking_data in tracking_numbers:
                                    tracking_info = {
                                        'source': 'shipstation',
                                        'tracking_number': tracking_data.get('tracking_number'),
                                        'carrier': tracking_data.get('carrier'),
                                        'service': tracking_data.get('service'),
                                        'ship_date': tracking_data.get('ship_date'),
                                        'order_id': processed_order.get('order_id'),
                                        'order_number': processed_order.get('order_number')
                                    }
                                    email_data['tracking_info'].append(tracking_info)
                            else:
                                log.info(f"No tracking numbers found for order {order_number}")
                        
            except Exception as e:
                log.error(f"Error processing order number {order_number}: {e}")
                continue
    
    # Strategy 2: Enhanced customer name search (if no orders found by unified search or order number)
    if not orders_found and customer_names:
        log.info("No orders found by order number, trying customer name search")
        
        for customer_name in customer_names:
            if not customer_name or len(customer_name.strip()) < 2:
                continue
            
            # Clean customer name for better matching
            clean_name = customer_name.strip()
            if clean_name.isdigit() or '@' in clean_name:
                continue
                
            log.info(f"Searching for orders by customer name: {clean_name}")
            try:
                # Use the existing get_orders_by_customer_name method
                customer_orders = shipstation_api.get_orders_by_customer_name(clean_name)
                if customer_orders:
                    log.info(f"Found {len(customer_orders)} orders for customer: {clean_name}")
                    orders_found = True
                    
                    # Process up to 5 most recent orders to provide comprehensive context
                    for order_data in customer_orders[:5]:
                        processed_order = shipstation_api.process_order_data(order_data)
                        
                        if processed_order:
                            unified_order_data = {
                                'source': 'shipstation',
                                'search_method': 'customer_name',
                                'search_value': clean_name,
                                'order_id': processed_order.get('order_id'),
                                'order_number': processed_order.get('order_number'),
                                'order_status': processed_order.get('order_status'),
                                'order_date': processed_order.get('order_date'),
                                'order_total': processed_order.get('order_total'),
                                'recipient_name': processed_order.get('recipient_name'),
                                'customer_email': processed_order.get('customer_email'),
                                'shipping_details': processed_order.get('shipping_details', {}),
                                'items': processed_order.get('items', []),
                                'ship_by_date': processed_order.get('ship_by_date'),
                                'unified_data': processed_order
                            }
                            
                            email_data['order_info'].append(unified_order_data)
                            
                            # Get tracking information
                            if processed_order.get('order_id'):
                                tracking_numbers = shipstation_api.get_tracking_numbers(processed_order['order_id'])
                                
                                for tracking_data in tracking_numbers:
                                    tracking_info = {
                                        'source': 'shipstation',
                                        'search_method': 'customer_name',
                                        'tracking_number': tracking_data.get('tracking_number'),
                                        'carrier': tracking_data.get('carrier'),
                                        'service': tracking_data.get('service'),
                                        'ship_date': tracking_data.get('ship_date'),
                                        'order_id': processed_order.get('order_id'),
                                        'order_number': processed_order.get('order_number')
                                    }
                                    email_data['tracking_info'].append(tracking_info)
                    
                    # Break after finding orders for the first valid customer name
                    break
                    
            except Exception as e:
                log.error(f"Error processing customer name {clean_name}: {e}")
                continue
    
    # Strategy 3: Try Pompeii3 API as additional fallback
    if not orders_found and order_numbers:
        log.info("Trying Pompeii3 API as fallback for order numbers")
        
        for order_number in order_numbers:
            if not order_number:
                continue
                
            try:
                pompeii_data = get_order_status_from_pompeii3(order_number, log)
                
                if pompeii_data and not pompeii_data.get('error'):
                    log.info(f"Found order {order_number} in Pompeii3 API")
                    
                    # Convert Pompeii3 data to unified format
                    unified_order_data = {
                        'source': 'pompeii3_api',
                        'order_id': pompeii_data.get('order_id_internal'),
                        'order_number': pompeii_data.get('order_number'),
                        'order_status': pompeii_data.get('order_status'),
                        'recipient_name': pompeii_data.get('recipient_name'),
                        'shipping_method': pompeii_data.get('shipping_method'),
                        'estimated_ship_date': pompeii_data.get('estimated_ship_date'),
                        'items': pompeii_data.get('items', []),
                        'production_percentage': pompeii_data.get('production_percentage'),
                        'tracking_link': pompeii_data.get('tracking_link'),
                        'unified_data': pompeii_data
                    }
                    
                    email_data['order_info'].append(unified_order_data)
                    
                    # Add tracking info if available
                    if pompeii_data.get('tracking_number'):
                        tracking_info = {
                            'source': 'pompeii3_api',
                            'tracking_number': pompeii_data.get('tracking_number'),
                            'tracking_link': pompeii_data.get('tracking_link'),
                            'order_number': pompeii_data.get('order_number')
                        }
                        email_data['tracking_info'].append(tracking_info)
                    
                    orders_found = True
                    
            except Exception as e:
                log.error(f"Error querying Pompeii3 API for order {order_number}: {e}")
                continue
    
    # Log final results
    if email_data.get('order_info'):
        log.info(f"Successfully processed {len(email_data['order_info'])} orders")
        log.info(f"Successfully processed {len(email_data.get('tracking_info', []))} tracking records")
    else:
        log.info("No orders found for this email")
        
    # Ensure the fields exist even if empty
    if 'order_info' not in email_data:
        email_data['order_info'] = []
    if 'tracking_info' not in email_data:
        email_data['tracking_info'] = []



def process_order_tracking_info_comprehensive(
    email_data: Dict[str, Any], 
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Comprehensive function to process order and tracking information using the original architecture.
    This function handles both explicit order numbers and tracking numbers from email entities.
    """
    log = logger or logging.getLogger(__name__)
    
    if not email_data.get('entities'):
        log.info("No entities in email_data to process for order/tracking.")
        return

    # Convert dict entities back to ExtractedEntity objects for function calls
    entities_list: List[ExtractedEntity] = []
    for entity_dict in email_data['entities']:
        if isinstance(entity_dict, dict) and 'type' in entity_dict and 'value' in entity_dict:
            entities_list.append(
                ExtractedEntity(
                    type=entity_dict['type'],
                    value=entity_dict['value'],
                    confidence=entity_dict.get('confidence', 0.9)
                )
            )
        elif isinstance(entity_dict, ExtractedEntity):
            entities_list.append(entity_dict)

    order_number_entities = [e for e in entities_list if e.type == "ORDER_NUMBER"]
    purchase_order_entities = [e for e in entities_list if e.type == "PURCHASE_ORDER_NUMBER"]
    
    all_orders_info = []

    if order_number_entities:
        log.info(f"Detected {len(order_number_entities)} ORDER_NUMBER entities. Retrieving order information.")
        try:
            regular_orders = get_order_info(order_number_entities, log)
            if regular_orders:
                all_orders_info.extend(regular_orders)
                log.info(f"Retrieved info for {len(regular_orders)} regular orders.")
        except Exception as order_error:
            log.error(f"Error processing regular order information: {order_error}", exc_info=True)

    if purchase_order_entities:
        log.info(f"Detected {len(purchase_order_entities)} PURCHASE_ORDER_NUMBER entities. Retrieving PO information.")
        try:
            po_orders = get_purchase_order_info(purchase_order_entities, log)
            if po_orders:
                all_orders_info.extend(po_orders)
                log.info(f"Retrieved info for {len(po_orders)} purchase orders.")
        except Exception as po_error:
            log.error(f"Error processing purchase order information: {po_error}", exc_info=True)

    # Enhanced customer name processing for when no orders are found
    customer_name_entities = [e for e in entities_list if e.type == "CUSTOMER_NAME"]
    if not all_orders_info and customer_name_entities:
        log.info(f"No orders found by order numbers. Trying customer name search with {len(customer_name_entities)} customer names.")
        
        # Initialize ShipStation client for customer search
        try:
            shipstation_client = ShipStationAPI(SHIPSTATION_API_KEY, SHIPSTATION_API_SECRET, logger=log)
            
            for customer_entity in customer_name_entities:
                customer_name = customer_entity.value
                if not customer_name or len(customer_name.strip()) < 2:
                    continue
                
                # Clean customer name for better matching
                clean_customer_name = customer_name.strip()
                if clean_customer_name.isdigit() or '@' in clean_customer_name:
                    continue
                    
                try:
                    customer_orders = shipstation_client.get_orders_by_customer_name(clean_customer_name)
                    if customer_orders:
                        log.info(f"Found {len(customer_orders)} orders for customer: {clean_customer_name}")
                        
                        # Convert to unified format
                        for order_data in customer_orders[:3]:  # Limit to 3 most recent
                            processed_order = shipstation_client.process_order_data(order_data)
                            if processed_order:
                                unified_order_data = create_unified_order_data(None, processed_order, log)
                                if unified_order_data:
                                    all_orders_info.append({
                                        'original_query_customer_name': clean_customer_name,
                                        'unified_data': unified_order_data,
                                        'shipstation_source_data': processed_order,
                                        'search_method': 'customer_name'
                                    })
                        break  # Found orders, stop searching
                        
                except Exception as e:
                    log.error(f"Error searching by customer name {clean_customer_name}: {e}")
                    continue
                    
        except Exception as e:
            log.error(f"Failed to initialize ShipStation for customer search: {e}")

    if all_orders_info:
        email_data['order_info'] = all_orders_info
        log.info(f"Updated email_data with combined info for {len(all_orders_info)} orders/POs.")
        
        # Collect all tracking numbers from all successfully fetched orders/POs
        all_tracking_entities_from_orders: List[ExtractedEntity] = []
        seen_tns_for_direct_api_call = set()

        for order_detail in all_orders_info:
            if order_detail and not order_detail.get('error'):
                # Unified data should have 'tracking_info_list'
                tracking_list_from_unified = order_detail.get('unified_data', {}).get('tracking_info_list', [])
                for track_item in tracking_list_from_unified:
                    tn = track_item.get('tracking_number')
                    if tn and tn not in seen_tns_for_direct_api_call:
                        all_tracking_entities_from_orders.append(
                            ExtractedEntity(type="TRACKING_NUMBER", value=tn, confidence=0.95)
                        )
                        seen_tns_for_direct_api_call.add(tn)
                
                # Fallback for Pompeii3 PO data
                if order_detail.get('is_purchase_order') and order_detail.get('pompeii3_source_data'):
                    tn = order_detail['pompeii3_source_data'].get('tracking_number')
                    if tn and tn not in seen_tns_for_direct_api_call:
                        all_tracking_entities_from_orders.append(
                            ExtractedEntity(type="TRACKING_NUMBER", value=tn, confidence=0.95)
                        )
                        seen_tns_for_direct_api_call.add(tn)

        if all_tracking_entities_from_orders:
            log.info(f"Found {len(all_tracking_entities_from_orders)} unique tracking numbers from fetched order details. Getting detailed tracking info.")
            try:
                detailed_tracking_info_list = get_tracking_info(all_tracking_entities_from_orders, log)
                if detailed_tracking_info_list:
                    email_data['tracking_info'] = detailed_tracking_info_list
                    log.info(f"Retrieved detailed tracking info for {len(detailed_tracking_info_list)} tracking numbers from orders.")
            except Exception as tracking_error:
                log.error(f"Error processing tracking info from orders: {tracking_error}", exc_info=True)
        else:
            log.info("No tracking numbers found associated with the fetched order details.")

    # If no order_info was found (or if orders had no tracking),
    # check for TRACKING_NUMBER entities directly extracted from email text.
    if not email_data.get('tracking_info'):
        tracking_number_entities_direct = [e for e in entities_list if e.type == "TRACKING_NUMBER"]
        # Filter out tracking numbers already processed via order lookup
        unprocessed_direct_tn_entities = [
            tn_entity for tn_entity in tracking_number_entities_direct 
            if tn_entity.value not in seen_tns_for_direct_api_call
        ]

        if unprocessed_direct_tn_entities:
            log.info(f"Detected {len(unprocessed_direct_tn_entities)} TRACKING_NUMBER entities directly in email text (not found via orders). Getting their info.")
            try:
                direct_tracking_info_list = get_tracking_info(unprocessed_direct_tn_entities, log)
                if direct_tracking_info_list:
                    email_data['tracking_info'] = email_data.get('tracking_info', []) + direct_tracking_info_list
                    log.info(f"Retrieved tracking info for {len(direct_tracking_info_list)} directly found tracking numbers.")
            except Exception as tracking_error:
                log.error(f"Error processing direct tracking number entities: {tracking_error}", exc_info=True)
        elif tracking_number_entities_direct:
            log.info("All directly found tracking numbers were already processed via order details.")
    
    log.info("Finished processing order and tracking information for email_data.")

