public static byte[] encodeInitialContextToken(InitialContextToken ctxToken, Codec codec) {
    try {
        Any any = ORB.init().create_any();
        InitialContextTokenHelper.insert(any, ctxToken);
        byte[] encodedToken = codec.encode_value(any);

        int innerLen = encodedToken.length + gssUpMechOidArray.length;

        int lenBytes;
        if (innerLen < 128) {
            lenBytes = 0;
        } else if (innerLen < 256) {
            lenBytes = 1;
        } else if (innerLen < 65536) {
            lenBytes = 2;
        } else if (innerLen < 16777216) {
            lenBytes = 3;
        } else {
            lenBytes = 4;
        }

        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream(2 + lenBytes + innerLen);
        baos.write(0x60);

        if (lenBytes == 0) {
            baos.write((byte) innerLen);
        } else {
            baos.write((byte) (lenBytes | 0x80));
            switch (lenBytes) {
                case 1:
                    baos.write((byte) innerLen);
                    break;
                case 2:
                    baos.write((byte) (innerLen >> 8));
                    baos.write((byte) innerLen);
                    break;
                case 3:
                    baos.write((byte) (innerLen >> 16));
                    baos.write((byte) (innerLen >> 8));
                    baos.write((byte) innerLen);
                    break;
                default:
                    baos.write((byte) (innerLen >> 24));
                    baos.write((byte) (innerLen >> 16));
                    baos.write((byte) (innerLen >> 8));
                    baos.write((byte) innerLen);
            }
        }

        baos.write(gssUpMechOidArray, 0, gssUpMechOidArray.length);
        baos.write(encodedToken, 0, encodedToken.length);

        return baos.toByteArray();
    } catch (Exception e) {
        return new byte[0];
    }
}