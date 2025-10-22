@Override
    public JSONObject toJsonObject() throws JSONException
    {
        JSONObject returnVal = super.toJsonObject();

        //Attachment Path...
        if(this.getAttachmentPath() != null)
        {
            returnVal.put(JSONMapping.ATTACHMENT_PATH,
                    this.getAttachmentPath());
        }

        //Attachment Data Base64...
        if(this.getAttachmentDataBase64() != null)
        {
            returnVal.put(JSONMapping.ATTACHMENT_DATA_BASE64,
                    this.getAttachmentDataBase64());
        }

        return returnVal;
    }