# Backend API reference

Base prefix: `/api/v1`. In development the interactive Swagger page is `/docs`, OpenAPI JSON `/openapi.json`, and ReDoc `/redoc`. These docs routes are disabled in `BCI_ENV=production`.

## `GET /api/v1/health`

Reports process health independently of model state.

```json
{"status":"ok","model_status":"not_configured","model_loaded":false}
```

`model_status` is `not_configured`, `loading`, `ready`, or `failed`. The endpoint returns 200 while the process is serving requests, including when the model is not ready.

## `GET /api/v1/model/info`

Reports configured/loaded state and adapter metadata where known. The default response identifies model fields as unknown and the input metadata as provisional:

```json
{
  "configured": false,
  "loaded": false,
  "status": "not_configured",
  "model_name": null,
  "model_version": null,
  "capabilities": null,
  "expected_input": {
    "contract_status": "provisional",
    "sampling_rate_hz": 250.0,
    "samples": 2500,
    "channels": ["FZ", "C3", "CZ", "C4"]
  },
  "metadata": null
}
```

Adapter metadata may include `model_name`, `model_version`, and `capabilities`; no model name, version, or accuracy is fabricated.

## `POST /api/v1/predict`

Delegates one request to the configured predictor. The current schema is provisional and based on the checked-in notebooks; the ML developer has not confirmed the serving modality.

Request requirements:

- `sampling_rate_hz`: exactly 250.0 Hz in the current provisional contract.
- `channels`: exactly `FZ`, `C3`, `CZ`, `C4` in column order.
- `samples`: exactly 2,500 rows, each exactly four finite numeric values.
- Extra top-level fields are rejected.
- Body size is bounded by `BCI_MAX_REQUEST_BYTES` (default 2 MB).

Schema sketch only (not a complete request body; a valid request must contain all 2,500 numeric sample rows):

```json
{
  "sampling_rate_hz": 250.0,
  "channels": ["FZ", "C3", "CZ", "C4"],
  "samples": [
    [324761.375, 321002.625, 335287.28125, 321417.90625],
    "... 2499 additional rows ..."
  ]
}
```

Success response:

```json
{"predicted_label":"Left","model_version":"<adapter version>","probabilities":null}
```

`probabilities` is null unless the adapter exposes valid class probabilities. They are not described as calibrated confidence by this API.

### Error responses

All API errors use an `error` envelope. Validation details include field locations and messages but omit submitted values.

| Status | Code | Meaning |
|---:|---|---|
| 413 | `request_too_large` | Body exceeds configured limit. |
| 422 | `validation_error` | Invalid JSON, missing field, wrong shape, channel order, sampling rate, non-finite value, or extra field. |
| 503 | `model_unavailable` | No model artifact/predictor is configured. |
| 503 | `model_load_failed` | Artifact or adapter could not initialize or is not ready. |
| 500 | `prediction_failed` | Inference operation raised an error. |
| 500 | `internal_error` | Unexpected internal/response serialization failure. |

Example 503:

```json
{"error":{"code":"model_unavailable","message":"Prediction model is not available"}}
```

No endpoint returns a fake prediction when the model is unavailable.
