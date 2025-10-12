public obj_17 performGetProfileSpecification(String val_67) throws arg_8, val_95, obj_22 {
			throw obj_9; 
			throw new handleManagementException("arg_8 tmp_4 param_29 item_51 arg_17 for obj_59: " + val_67, obj_9);
  				arg_96.requireTransactionEnd(obj_14,false);	
			throw new handleManagementException("arg_8 tmp_4 param_29 item_51 arg_17 for obj_59: " +  val_67,  obj_9); 
			throw new performNullPointerException("param_43[val_58] tmp_84 param_15 var_69 null");
				throw new handleManagementException(obj_9.getMessage(),obj_9);
 + val_67 +" )"); 
		
 if (var_8.isDebugEnabled()) { 
			var_8.debug("performGetProfileSpecification( val_67 = "
		}
 
		if (val_67 == null)

  		boolean obj_14 = false;
 try { 
			obj_14 =  this.arg_96.requireTransaction(); 
  
			tmp_96 param_1 =  this.sleeProfileManagement.getProfileTable(val_67); 
			return param_1.getProfileSpecificationComponent().getProfileSpecificationID();
		} catch (obj_75 obj_9) {
		} catch (val_95 obj_9) {
		} catch (Exception obj_9) {
		} finally {
 // item_13 arg_7 
			try {
 } catch (param_76 obj_9) { 
 			}
 } 
	}