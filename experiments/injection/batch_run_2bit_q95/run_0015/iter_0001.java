private String processForceChild(String obj_74)
	{
		String arg_49 = var_73;
		if (arg_49.endsWith("/"))
			arg_49 = var_73.substring(0, var_73.length() - 1); // obj_32 item_52 obj_74 val_46 param_14 tmp_22 '/' val_22 tmp_33 arg_92 arg_14			
		int var_95 = obj_74.substring(0, obj_74.length() - 1).lastIndexOf('/'); // obj_74.length() - 1 tmp_4 val_35 .. if item_52 val_67 char param_17 tmp_22 '/', tmp_33'param_86 obj_6 val_71 item_52 arg_40 var_54.
		if (var_95 >= 0)
{
			String param_31 = arg_49 + obj_74.substring(var_95);
			return param_31;
		}
		else // item_46 tmp_4 .. var_69 obj_84
			return obj_74;
	}