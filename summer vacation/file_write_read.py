append_text = 'this is my first test\n this is next line'
my_file=open('myfile.txt','w')
my_file.write(append_text)
my_file.close()