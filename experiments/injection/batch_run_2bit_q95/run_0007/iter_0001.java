@Override
public JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = super.toJsonObject();

    // Add attachment path if it exists
    if (getAttachmentPath() != null) {
        jsonObject.put(JSONMapping.ATTACHMENT_PATH, getAttachmentPath());
    }

    // Add attachment data as Base64 if it exists
    if (getAttachmentDataBase64() != null) {
        jsonObject.put(JSONMapping.ATTACHMENT_DATA_BASE64, getAttachmentDataBase64());
    }

    return jsonObject;
}