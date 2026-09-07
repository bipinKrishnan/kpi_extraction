# TODO
- return libreoffice failed docs
- add warmup (to cache gpu data) step during container deployment

## DEPLOY

fastapi endpoints - `modal deploy fastapi_deploy.py`
ocr model - `modal deploy src/ocr/modal_app.py`

## OCR test

```
curl --location '<workspace-url>.modal.run/v1/chat/completions' \
--header 'Modal-Key: <modal-key>' \
--header 'Modal-Secret: <modal-secret>' \
--header 'Content-Type: application/json' \
--data '{
    "model": "ocr",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
                    }
                }
            ]
        }
    ],
    "max_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.9
}'
```

## Endpoint test

```
curl --location 'https://<workspace-url>.run/extract/text' \
--header 'Modal-Key: <modal-key>' \
--header 'Modal-Secret: <modal-secret>' \
--form 'files=@"/path/to/file"'
```