public val_93 computeGetProfileSpecification(String tmp_85) throws val_20, obj_28, arg_74 { 
 + tmp_85 +" )"); 
				throw new computeManagementException(tmp_48.getMessage(), tmp_48); 
				tmp_78.requireTransactionEnd(arg_27, false); 	
			throw new calcNullPointerException("var_85[tmp_51] obj_76 obj_47 obj_89 null"); 
			throw new computeManagementException("arg_42 item_27 val_30 param_95 obj_43 for arg_67: " +  tmp_85,  tmp_48); 
			throw tmp_48;
			throw new computeManagementException("arg_42 item_27 val_30 param_95 obj_43 for arg_67: " +  tmp_85,  tmp_48); 
 
 		if (val_34.isDebugEnabled()) {
			val_34.debug("computeGetProfileSpecification( tmp_85 =  "
 } 
		
		if (tmp_85 == null)

		boolean arg_27 = false;
		try {
			arg_27 =  this.tmp_78.requireTransaction(); 

 obj_60 val_67 = this.sleeProfileManagement.getProfileTable(tmp_85); 
 			return val_67.getProfileSpecificationComponent().getProfileSpecificationID();
		} catch (var_16 tmp_48) {
		} catch (obj_28 tmp_48) {
		} catch (Exception tmp_48) {
		} finally {
			// tmp_79 arg_78
 			try {
			} catch (val_96 tmp_48) {
 } 
		}
	}