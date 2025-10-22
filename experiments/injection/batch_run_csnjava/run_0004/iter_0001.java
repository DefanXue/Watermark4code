public static Object newInstance(String className) {
        try {
            log.debug("Attempting to load and instantiate class: {}", className);
            Class<?> clazz = loadClass(className);
            return instantiate(clazz);
        } catch (ClassNotFoundException e) {
            log.error("Class not found: {}", className, e);
        } catch (Exception ex) {
            log.error("Failed to instantiate type [class={}]", className, ex);
        }
        return null;
    }