public void pushComponentCounters(String name, PerformanceCountersHolder holder) {
      if (snapshot == null)
 return;

 final PerformanceCountersHolder countersHolder = snapshot.countersByComponent.get(name);  // Main business logic

 if (countersHolder != null)
{
      countersHolder.pushData(holder);  // Main business logic
 }  // Check boundary conditions
 }