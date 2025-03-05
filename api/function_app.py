import azure.functions as func
import requests
import json
import os
import traceback
import logging
import time
import sys
import platform
from datetime import datetime
from typing import Dict, Any

app = func.FunctionApp()

# Configure basic logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Required environment variables
required_vars = [
    "PROMPTFLOW_ENDPOINT",
    "PROMPTFLOW_KEY",
    "AZURE_AI_SEARCH_ENDPOINT",
    "AZURE_AI_SEARCH_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "FUNCTION_KEY"
    # Removed APPLICATIONINSIGHTS_CONNECTION_STRING
]

def get_system_info() -> Dict[str, Any]:
    """Get basic system information for debugging"""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "env_vars_present": [k for k in required_vars if os.environ.get(k) is not None],
    }

def get_env_vars_with_retry(max_retries=3, retry_delay=1) -> Dict[str, str]:
    """Get environment variables with retry mechanism for cold starts"""
    for attempt in range(max_retries):
        try:
            env_vars = {var: os.environ.get(var) for var in required_vars}
            missing_vars = [var for var, value in env_vars.items() if not value]
            
            if not missing_vars:
                return env_vars
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            pass
    
    error_msg = f"Failed to load environment variables after {max_retries} attempts: {missing_vars}"
    raise ValueError(error_msg)

def create_error_response(error_type: str, details: str, status_code: int, request_id: str, additional_info: Dict[str, Any] = None) -> func.HttpResponse:
    """Create a standardized error response with debug information"""
    error_body = {
        "error": error_type,
        "details": details,
        "debug_info": {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            **get_system_info()
        }
    }
    
    if additional_info:
        error_body["debug_info"].update(additional_info)
    
    headers = {
        "X-Debug-RequestId": request_id,
        "X-Debug-ErrorType": error_type,
        "X-Debug-Timestamp": error_body["debug_info"]["timestamp"],
        "Access-Control-Expose-Headers": "X-Debug-RequestId, X-Debug-ErrorType, X-Debug-Timestamp"
    }
    
    return func.HttpResponse(
        json.dumps(error_body),
        mimetype="application/json",
        status_code=status_code,
        headers=headers
    )

@app.route(route="chat", methods=["POST"])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    request_id = f"req_{int(time.time())}_{os.urandom(4).hex()}"
    debug_info = {"request_start_time": datetime.now().isoformat()}
    
    try:
        # Parse request body
        try:
            req_body = req.get_json()
            debug_info["request_body_size"] = len(json.dumps(req_body))
        except ValueError as e:
            return create_error_response(
                "Invalid request",
                "Failed to parse request body as JSON",
                400,
                request_id,
                {"parse_error": str(e)}
            )

        user_query = req_body.get('query')
        chat_history = req_body.get('chats', [])
        debug_info.update({
            "query_length": len(user_query) if user_query else 0,
            "chat_history_length": len(chat_history)
        })

        if not user_query:
            return create_error_response(
                "No query provided",
                "The query parameter is required",
                400,
                request_id,
                debug_info
            )

        # Get environment variables
        try:
            env_vars = get_env_vars_with_retry()
            debug_info["env_vars_loaded"] = True
        except ValueError as e:
            return create_error_response(
                "Configuration error",
                str(e),
                503,
                request_id,
                debug_info
            )

        # Prepare prompt flow request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {env_vars["PROMPTFLOW_KEY"]}',
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
        }

        request_body = {
            "azure_ai_search_endpoint": env_vars["AZURE_AI_SEARCH_ENDPOINT"],
            "azure_ai_search_key": env_vars["AZURE_AI_SEARCH_KEY"],
            "azure_openai_key": env_vars["AZURE_OPENAI_KEY"],
            "azure_openai_endpoint": env_vars["AZURE_OPENAI_ENDPOINT"],
            "function_key": env_vars["FUNCTION_KEY"],
            "query": user_query,
            "chat_history": chat_history
        }

        # Make prompt flow request
        try:
            start_time = time.time()
            response = requests.post(
                env_vars["PROMPTFLOW_ENDPOINT"],
                headers=headers,
                json=request_body,
                timeout=180
            )
            request_time = time.time() - start_time
            debug_info["prompt_flow_time"] = request_time
            debug_info["prompt_flow_status"] = response.status_code
            
            response_text = response.text
            
            # Log the entire response for debugging
            logger.info(f"Prompt flow response for request {request_id}: {response_text[:1000]}...")
            
            # Specifically check for backend call failure
            if 'Backend call failure' in response_text:
                logger.error(f"Backend call failure detected for request {request_id}")
                return create_error_response(
                    "Backend call failure",
                    "The prompt flow service encountered a backend failure",
                    502,  # Bad Gateway
                    request_id,
                    {
                        **debug_info,
                        "response_preview": response_text[:500],
                        "error_type": "backend_call_failure"
                    }
                )
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            return create_error_response(
                "Request timeout",
                "The request to the prompt flow service timed out",
                504,
                request_id,
                {**debug_info, "timeout_after": 180}
            )
        except requests.exceptions.RequestException as e:
            error_details = {
                **debug_info,
                "error_type": type(e).__name__,
                "prompt_flow_error": str(e)
            }
            
            if hasattr(e, 'response') and e.response is not None:
                error_details.update({
                    "prompt_flow_status": e.response.status_code,
                    "prompt_flow_response_preview": e.response.text[:500]
                })
            
            return create_error_response(
                "Prompt flow service error",
                str(e),
                502,
                request_id,
                error_details
            )

        # Parse prompt flow response
        try:
            response_data = response.json()
            debug_info["response_size"] = len(json.dumps(response_data))
            logger.info(f"Successfully parsed JSON response for request {request_id}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for request {request_id}: {str(e)}")
            return create_error_response(
                "Invalid response",
                "Failed to parse response from prompt flow service",
                502,
                request_id,
                {
                    **debug_info,
                    "response_preview": response.text[:500],
                    "parse_error": str(e)
                }
            )

        # Add debug headers to success response
        headers = {
            "X-Debug-RequestId": request_id,
            "X-Debug-Time": str(debug_info["prompt_flow_time"]),
            "X-Debug-Timestamp": datetime.now().isoformat(),
            "Access-Control-Expose-Headers": "X-Debug-RequestId, X-Debug-Time, X-Debug-Timestamp"
        }

        # Success case
        logger.info(f"Returning successful response for request {request_id}")
        return func.HttpResponse(
            json.dumps({
                "data": response_data,
                "debug_info": debug_info
            }),
            mimetype="application/json",
            headers=headers
        )

    except Exception as e:
        logger.error(f"Unhandled exception in chat function for request {request_id}: {str(e)}")
        error_info = {
            "request_id": request_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "stack_trace": traceback.format_exc(),
            **debug_info
        }
        
        return create_error_response(
            "Internal server error",
            str(e),
            500,
            request_id,
            error_info
        )