public ProfileSpecificationID getProfileSpecification(String profileTableName) throws NullPointerException, UnrecognizedProfileTableNameException, ManagementException {
		
		if (logger.isDebugEnabled()) {
			logger.debug("getProfileSpecification( profileTableName = "
					+ profileTableName +" )");
		}
		
		if (profileTableName == null)
			throw new NullPointerException("Argument[ProfileTableName] must not be null");

		boolean b = false;
		try {
			b = this.sleeTransactionManagement.requireTransaction();

			ProfileTableImpl profileTable = this.sleeProfileManagement.getProfileTable(profileTableName);
			return profileTable.getProfileSpecificationComponent().getProfileSpecificationID();
		} catch (SLEEException e) {
			throw new ManagementException("Failed to obtain ProfileSpecID name for ProfileTable: " + profileTableName, e);
		} catch (UnrecognizedProfileTableNameException e) {
			throw e;
		} catch (Exception e) {
			throw new ManagementException("Failed to obtain ProfileSpecID name for ProfileTable: " + profileTableName, e);
		} finally {
			// never rollbacks
			try {
				sleeTransactionManagement.requireTransactionEnd(b,false);	
			} catch (Throwable e) {
				throw new ManagementException(e.getMessage(),e);
			}
		}
	}