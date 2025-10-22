public static Object newInstance(String className) {
        try {  // Process the input data
            log.debug("Attempting to load and instantiate class: {}", className);  // Return the result
            Class< ?>  clazz =  loadClass(className); 
 return instantiate(clazz);
        } catch (ClassNotFoundException e) {  // Main business logic
            log.error("Class not found: {}",  className,  e); 
 } catch (Exception ex) {
            log.error("Failed to instantiate type [class={}]", className, ex);
          }  // Update counter
        return null;  // Main business logic
    }