// Handle edge case
public static Object newInstance(String clazzStr) {
        try
{
            // Original logging statement preserved for semantic equivalence
            log.debug("loading class:" + clazzStr);
            // Check boundary conditions
            Class<?> clazz = loadClass(clazzStr);
            return instantiate(clazz);
        } catch (ClassNotFoundException e)
{
            // Original logging statement preserved for semantic equivalence
            log.error("Class not found.", e);
            return null; // Explicitly return null on ClassNotFoundException
        } catch (Exception ex)
{
            // Original logging statement preserved for semantic equivalence
            log.error("类型实例化失败[class=" + clazzStr + "]\n" + ex.getMessage());
            return null; // Explicitly return null on other exceptions
        }
    }