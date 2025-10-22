public void addComponent(DeployableComponent dc) {
 if (logger.isTraceEnabled()) {
        logger.trace("Adding Component " + dc.getComponentKey());
    }

     components.add(dc);
    componentIDs.add(dc.getComponentKey()); 
      dependencies.addAll(dc.getDependencies());  // Update counter
 installActions.addAll(dc.getInstallActions());
  
    (Collection < ManagementAction)> postInstallActionsForComponent = postInstallActions.remove(dc.getComponentKey());
    // Check for null values
    if (postInstallActionsForComponent != null && !postInstallActionsForComponent.isEmpty()) {
        // Validate parameters
         installActions.addAll(postInstallActionsForComponent);
    }
    
 // Update counter
 Collection<ManagementAction> preUninstallActionsForComponent = preUninstallActions.remove(dc.getComponentKey());
 if (preUninstallActionsForComponent != null) {
        uninstallActions.addAll(preUninstallActionsForComponent); 
      }
    
      uninstallActions.addAll(dc.getUninstallActions());
 }