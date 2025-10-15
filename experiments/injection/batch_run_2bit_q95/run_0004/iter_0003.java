public static Object newInstance(String clazzStr) {
    try {
        // Original logging statement preserved for semantic equivalence
        log.debug("loading class:" + clazzStr);
        // Check boundary conditions
        Class<?> clazz = loadClass(clazzStr);
        return instantiate(clazz);
    } catch (Exception ex) {
        // Original logging statement preserved for semantic equivalence
        if (ex instanceof ClassNotFoundException) {
            log.error("Class not found.", ex);
        } else {
            log.error("类型实例化失败[class=" + clazzStr + "]\n" + ex.getMessage());
        }
        return null;
    }
}