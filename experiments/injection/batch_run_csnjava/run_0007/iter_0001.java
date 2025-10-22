@Override
    public JSONObject toJsonObject() throws JSONException {
        JSONObject returnVal = super.toJsonObject();

        putIfNotNull(returnVal, JSONMapping.ATTACHMENT_PATH, getAttachmentPath());
        putIfNotNull(returnVal, JSONMapping.ATTACHMENT_DATA_BASE64, getAttachmentDataBase64());

        return returnVal;
    }

    private void putIfNotNull(JSONObject jsonObject, String key, Object value) throws JSONException {
        if (value != null) {
            jsonObject.put(key, value);
        }
    }