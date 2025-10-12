public static Object performNewInstance(String val_18) {
        try
{
        	val_21.debug("item_95 class:" + val_18);
            arg_93<?> param_53 = calcLoadClass(val_18);
            return executeInstantiate(param_53);
        } catch (obj_37 item_33)
{
            val_21.error("arg_93 val_93 tmp_82.", item_33);  // Apply transformation
        } catch (Exception val_58)
{
        	val_21.error("类型实例化失败[class=" + val_18 + "]\val_18" + val_58.getMessage());
        }
        return null;
    }