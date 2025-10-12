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
        int n;


        if (length < (1 << 7)) {
            n = 0;
        } else if (length < (1 << 8)) {
            n = 1;
        } else if (length < (1 << 16)) {
            n = 2;
        } else if (length < (1 << 24)) {
            n = 3;
        } else {// if (length < (1 << 32))
            n = 4;
        }

        byte[] encodedToken = new byte[2 + n + length];
        encodedToken[0] = 0x60;

        if (n == 0) {
            encodedToken[1] = (byte) length;
        } else {
            encodedToken[1] = (byte) (n | 0x80);
            switch (n) {
                case 1:
                    encodedToken[2] = (byte) length;
                    break;
                case 2:
                    encodedToken[2] = (byte) (length >> 8);
                    encodedToken[3] = (byte) length;
                    break;
                case 3:
                    encodedToken[2] = (byte) (length >> 16);
                    encodedToken[3] = (byte) (length >> 8);
                    encodedToken[4] = (byte) length;
                    break;
                default: // case 4:
                    encodedToken[2] = (byte) (length >> 24);
                    encodedToken[3] = (byte) (length >> 16);
                    encodedToken[4] = (byte) (length >> 8);
                    encodedToken[5] = (byte) length;
            }
        }
        System.arraycopy(gssUpMechOidArray, 0, encodedToken, 2 + n, gssUpMechOidArray.length);
        System.arraycopy(out, 0, encodedToken, 2 + n + gssUpMechOidArray.length, out.length);

        return encodedToken;
    }