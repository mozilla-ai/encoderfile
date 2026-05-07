# Multipart OpenAPI Service Example

This document provides examples of how to interact with the multipart file upload and prediction endpoint.

## Endpoint Overview

- **POST /predict/multipart** - Submit a JSON payload with binary file attachments
- **GET /predict/multipart/openapi.json** - Retrieve the OpenAPI specification

## Example 1: cURL with Two Image Files

```bash
curl -X POST http://localhost:8080/predict/multipart \
  -F "payload={\"model_version\": \"1.0\", \"threshold\": 0.8}" \
  -F "files=@/path/to/image1.png" \
  -F "files=@/path/to/image2.jpg"
```

### Request Body (multipart/form-data)

```
--boundary_123abc456def
Content-Disposition: form-data; name="payload"
Content-Type: application/json

{"model_version": "1.0", "threshold": 0.8}
--boundary_123abc456def
Content-Disposition: form-data; name="files"; filename="image1.png"
Content-Type: image/png

<binary PNG data here>
--boundary_123abc456def
Content-Disposition: form-data; name="files"; filename="image2.jpg"
Content-Type: image/jpeg

<binary JPG data here>
--boundary_123abc456def--
```

### Response

```json
{
  "payload": {
    "model_version": "1.0",
    "threshold": 0.8
  },
  "attachment_count": 2,
  "attachments": [
    {
      "file_name": "image1.png",
      "content_type": "image/png",
      "size_bytes": 45230
    },
    {
      "file_name": "image2.jpg",
      "content_type": "image/jpeg",
      "size_bytes": 52104
    }
  ]
}
```

## Example 2: Python Requests Library

```python
import requests
import json

url = "http://localhost:8080/predict/multipart"

# Prepare the payload
payloaquest Body (multipart/form-data)

```
--boundary_xyz789pqr012
Content-Disposition: form-data; name="payload"; filename="payload.json"
Content-Type: application/json

{"model_version": "1.0", "threshold": 0.8, "batch_id": "batch_12345"}
--boundary_xyz789pqr012
Content-Disposition: form-data; name="files"; filename="image1.png"
Content-Type: image/png

<binary PNG data>
--boundary_xyz789pqr012
Content-Disposition: form-data; name="files"; filename="image2.jpg"
Content-Type: image/jpeg

<binary JPG data>
--boundary_xyz789pqr012
Content-Disposition: form-data; name="files"; filename="document.pdf"
Content-Type: application/pdf

<binary PDF data>
--boundary_xyz789pqr012--
```

### Red = {
    "model_version": "1.0",
    "threshold": 0.8,
    "batch_id": "batch_12345"
}

# Prepare files
files = [
    ("payload", ("payload.json", json.dumps(payload), "application/json")),
    ("files", ("image1.png", open("image1.png", "rb"), "image/png")),
    ("files", ("image2.jpg", open("image2.jpg", "rb"), "image/jpeg")),
    ("files", ("document.pdf", open("document.pdf", "rb"), "application/pdf")),
]

# Send the request
response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
```

### Response

```json
{
  "payload": {
    "model_version": "1.0",
    "threshold": 0.8,
    "batch_id": "batch_12345"
  },
  "attachment_count": 3,
  "attachments": [
    {
      "file_name": "image1.png",
      "content_type": "image/png",
      quest Body (multipart/form-data)

```
--boundary_webkit_abc123
Content-Disposition: form-data; name="payload"

{"model_version":"1.0","threshold":0.8,"inference_id":"inf_abc123"}
--boundary_webkit_abc123
Content-Disposition: form-data; name="files"; filename="photo1.jpg"
Content-Type: image/jpeg

<binary JPG data>
--boundary_webkit_abc123
Content-Disposition: form-data; name="files"; filename="photo2.jpg"
Content-Type: image/jpeg

<binary JPG data>
--boundary_webkit_abc123--
```

### Re"size_bytes": 45230
    },
    {
      "file_name": "image2.jpg",
      "content_type": "image/jpeg",
      "size_bytes": 52104
    },
    {
      "file_name": "document.pdf",
      "content_type": "application/pdf",
      "size_bytes": 128512
    }
  ]
}
```

## Example 3: JavaScript Fetch API

```javascript
const payload = {
  model_version: "1.0",
  threshold: 0.8,
  inference_id: "inf_abc123"
};

const formData = new FormData();

// Add the JSON payload as a form field
formData.append("payload", JSON.stringify(payload));

// Add multiple binary files
const imageFile1 = document.getElementById("imageInput1").files[0];
const imageFile2 = document.getElementById("imageInput2").files[0];

formData.append("files", imageFile1);
formData.append("files", imageFile2);

// Make the request
const response = await fetch("http://localhost:8080/predict/multipart", {
  method: "POST",
  body: formData
});

const result = await response.json();
console.log("Success:", result);
```

### Response

```json
{
  "payload": {
    "model_version": "1.0",
    "threshold": 0.8,
    "inference_id": "inf_abc123"
  },
  "attachment_count": 2,
  "attachments": [
    {
      "file_name": "photo1.jpg",
      "content_type": "image/jpeg",
      "size_bytes": 245120
    },
    {
      "file_name": "photo2.jpg",
      "content_type": "image/jpeg",
      "size_bytes": 187904
    }
  ]
}
```

## Example 4: Error Handling

### Missing Payload

If the request is sent without a `payload` form field:

```bash
curl -X POST http://localhost:8080/predict/multipart \
  -F "files=@/path/to/image.png"
```

**Response (422 Unprocessable Entity):**

```
missing required multipart field 'payload'
```

### Invalid JSON in Payload

If the payload field contains invalid JSON:

```bash
curl -X POST http://localhost:8080/predict/multipart \
  -F "payload=not valid json" \
  -F "files=@/path/to/image.png"
```

**Response (422 Unprocessable Entity):**

```
invalid json in 'payload' field
```

### Malformed Multipart Body

If the multipart encoding is corrupted:

**Response (400 Bad Request):**

```
multipart parse error: [error details]
```

## Request Parts Specification

### Required: `payload` Part

- **Name**: `payload` (exactly one)
- **Content-Type**: `application/json` (recommended)
- **Content**: Valid JSON object or array

### Optional: `files` Parts

- **Name**: `files` (zero or more)
- **Content-Type**: Any MIME type (e.g., `image/png`, `application/pdf`)
- **Content**: Binary data
- **Filename**: Optional but recommended (used in response metadata)

## Response Structure

```json
{
  "payload": "...",                 // Echo of the submitted JSON payload
  "attachment_count": 3,            // Number of files attached
  "attachments": [                  // Metadata for each file
    {
      "file_name": "...",           // Original filename if provided, null otherwise
      "content_type": "...",        // MIME type if provided, null otherwise
      "size_bytes": 12345           // File size in bytes
    }
  ]
}
```

## HTTP Status Codes

| Status | Meaning | Condition |
|--------|---------|-----------|
| 200 | OK | Request processed successfully |
| 400 | Bad Request | Malformed multipart body |
| 422 | Unprocessable Entity | Missing `payload` or invalid JSON |

## OpenAPI Specification

To retrieve the OpenAPI specification for this endpoint:

```bash
curl -X GET http://localhost:8080/predict/multipart/openapi.json
```

This returns a machine-readable OpenAPI 3.0 document describing the endpoint.
