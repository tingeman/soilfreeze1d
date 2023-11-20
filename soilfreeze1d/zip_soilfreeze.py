from zipfile import ZipFile

# Unlink / delete zipfile first???

zipObj = ZipFile('soilfreeze1d_model.zip', 'w')
zipObj.write('soilfreeze1d.py')
zipObj.write('model1.py')
zipObj.write('model1_nug_vs_ug.py')
zipObj.write('model2.py')
zipObj.write('model2b.py')
zipObj.write('model3.py')
zipObj.write('model3_nug.py')
zipObj.write('model3_spinup.py')
zipObj.write('model3_embankment.py')
zipObj.write('model4.py')
zipObj.close()