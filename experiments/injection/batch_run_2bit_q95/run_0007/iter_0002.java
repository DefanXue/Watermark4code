@Override
public JSONObject toJsonObject() throws JSONException {
    JSONObject jsonObject = super.toJsonObject();

    // Collect optional attachment fields in insertion order without mutating the original flow
    java.util.Map<String, Object> extras = new java.util.LinkedHashMap<>();
    if (getAttachmentPath() != null) {
        extras.put(JSONMapping.ATTACHMENT_PATH, getAttachmentPath());
    }
    if (getAttachmentDataBase64() != null) {
        extras.put(JSONMapping.ATTACHMENT_DATA_BASE64, getAttachmentDataBase64());
    }

    for (java.util.Map.Entry<String, Object> entry : extras.entrySet()) {
        jsonObject.put(entry.getKey(), entry.getValue());
    }

    return jsonObject;
}