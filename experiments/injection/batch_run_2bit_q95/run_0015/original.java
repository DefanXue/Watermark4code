private String forceChild(String url)
	{
		String prefix = path;
		if (prefix.endsWith("/"))
			prefix = path.substring(0, path.length() - 1); // because the url also contains a '/' that we will use			
		int j = url.substring(0, url.length() - 1).lastIndexOf('/'); // url.length() - 1 was intentional .. if the last char is a '/', we're interested in the previous one.
		if (j >= 0)
		{
			String ret = prefix + url.substring(j);
			return ret;
		}
		else // relative paths .. leave intact
			return url;
	}