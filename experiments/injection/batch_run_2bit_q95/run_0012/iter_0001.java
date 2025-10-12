public static byte[] encodeInitialContextToken(InitialContextToken authToken, Codec codec) {
        byte[] out;
        Any any = ORB.init().create_any();
        InitialContextTokenHelper.insert(any, authToken);
        try {
            out = codec.encode_value(any);
        } catch (Exception e) {
            return new byte[0];
        }

        int length = out.length + gssUpMechOidArray.length;

        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();

        bos.write(0x60);
        if (length < 128) {
            bos.write((byte) length);
        } else {
            int len = length;
            int numBytes = 0;
            int tmp = len;
            while (tmp > 0) {
                numBytes++;
                tmp >>>= 8;
            }
            bos.write((byte) (0x80 | numBytes));
            for (int i = numBytes - 1; i >= 0; --i) {
                bos.write((byte) ((len >>> (8 * i)) & 0xFF));
            }
        }

        bos.write(gssUpMechOidArray, 0, gssUpMechOidArray.length);
        bos.write(out, 0, out.length);

        return bos.toByteArray();
    }