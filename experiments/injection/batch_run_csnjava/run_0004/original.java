public static Object newInstance(String clazzStr) {
        try {
        	log.debug("loading class:" + clazzStr);
            Class<?> clazz = loadClass(clazzStr);
            return instantiate(clazz);
        } catch (ClassNotFoundException e) {
            log.error("Class not found.", e);
        } catch (Exception ex) {
        	log.error("类型实例化失败[class=" + clazzStr + "]\n" + ex.getMessage());
        }
        return null;
    }