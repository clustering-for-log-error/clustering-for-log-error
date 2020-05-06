import pandas as pd
import numpy as np 
from env import user, password, host

def get_db_url(dbname) -> str:
    url = 'mysql+pymysql://{}:{}@{}/{}'
    return url.format(user, password, host, dbname)

def get_zillow_data():
   '''
   Pulls in data from SQL server. 
   Only columns without large amounts of nulls where lat and long are not null
   And where bed and bath count are greater than 0
   ''' 
   query = '''
   SELECT properties_2017.parcelid, bathroomcnt, bedroomcnt, buildingqualitytypeid, calculatedfinishedsquarefeet,  
finishedsquarefeet12, fips, fullbathcnt, latitude, longitude, lotsizesquarefeet, propertycountylandusecode,     
rawcensustractandblock, regionidcity, regionidcounty, regionidzip, roomcnt, unitcnt, yearbuilt, structuretaxvaluedollarcnt,taxvaluedollarcnt, assessmentyear, landtaxvaluedollarcnt, taxamount,                      
predictions_2017.logerror,                       
predictions_2017.transactiondate,                
heatingorsystemdesc,            
propertylandusedesc
FROM properties_2017
JOIN predictions_2017 USING(parcelid)
LEFT JOIN airconditioningtype USING(airconditioningtypeid)
LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
LEFT JOIN buildingclasstype USING(buildingclasstypeid)
LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
LEFT JOIN propertylandusetype USING(propertylandusetypeid)
LEFT JOIN storytype USING(storytypeid)
LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
WHERE latitude IS NOT NULL
AND longitude IS NOT NULL
AND bathroomcnt > 0
AND bedroomcnt > 0
   '''
   df = pd.read_sql(query, get_db_url('zillow'))
   return df