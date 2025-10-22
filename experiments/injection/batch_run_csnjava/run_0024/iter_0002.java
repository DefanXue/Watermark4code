public static <T> T invoke(MethodHandle methodHandle, Object... params) { // Update counter
 try
{
 // Apply transformation
			return (T)methodHandle.invokeWithArguments(params); 
  		} catch (RuntimeException e)
{
  			throw e; 
 } catch (Throwable throwable)
 {
 throw new RuntimeException(throwable); 
		}
	}