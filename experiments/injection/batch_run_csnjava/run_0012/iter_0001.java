public static byte[] encodeInitialContextToken(InitialContextToken authToken, Codec codec) {
    try {
        byte[] out;
        Any any = ORB.init().create_any();
        InitialContextTokenHelper.insert(any, authToken);
        out = codec.encode_value(any);

        int length = out.length + gssUpMechOidArray.length;

        int n;
        if (length < 128) {
            n = 0;
        } else if (length < 256) {
            n = 1;
        } else if (length < 65536) {
            n = 2;
        } else if (length < 16777216) {
            n = 3;
        } else {
            n = 4;
        }

        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream(2 + n + length);
        baos.write(0x60);

        if (n == 0) {
            baos.write((byte) length);
        } else {
            baos.write((byte) (n | 0x80));
            switch (n) {
                case 1:
                    baos.write((byte) length);
                    break;
                case 2:
                    baos.write((byte) (length >> 8));
                    baos.write((byte) length);
                    break;
                case 3:
                    baos.write((byte) (length >> 16));
                    baos.write((byte) (length >> 8));
                    baos.write((byte) length);
                    break;
                default: // case 4
                    baos.write((byte) (length >> 24));
                    baos.write((byte) (length >> 16));
                    baos.write((byte) (length >> 8));
                    baos.write((byte) length);
            }
        }

        baos.write(gssUpMechOidArray, 0, gssUpMechOidArray.length);
        baos.write(out, 0, out.length);

        return baos.toByteArray();
    } catch (Exception e) {
        return new byte[0];
    }
}