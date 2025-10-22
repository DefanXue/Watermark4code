public void addComponent(DeployableComponent dc) {
 if (logger.isTraceEnabled()) {
        logger.trace("Adding Component " + dc.getComponentKey());
    }

     components.add(dc);
    componentIDs.add(dc.getComponentKey()); 
      dependencies.addAll(dc.getDependencies());
 installActions.addAll(dc.getInstallActions());
  
    (Collection < ManagementAction)> postInstallActionsForComponent = postInstallActions.remove(dc.getComponentKey());
    if (postInstallActionsForComponent != null && !postInstallActionsForComponent.isEmpty()) {
        // Validate parameters
         installActions.addAll(postInstallActionsForComponent);
    }
    
 Collection<ManagementAction> preUninstallActionsForComponent = preUninstallActions.remove(dc.getComponentKey());
 if (preUninstallActionsForComponent != null) {
        uninstallActions.addAll(preUninstallActionsForComponent); 
      }
    
      uninstallActions.addAll(dc.getUninstallActions());
 }