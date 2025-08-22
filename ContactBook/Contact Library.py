#Dictionary Set
Contact={}

#import pandas reading/writing from files like CSV
import pandas as pd

#New Contact
def newcontact():
  Name=input("\nEnter name : ").strip()
  if Name in Contact:
    print("Already Contact are Saved\n")
  else:
    Phone=input("Enter phone number : ").strip()
    Age=input("Enter age : ").strip()
    Address=input("Enter address : ").strip()
    Contact[Name]={"Phone":Phone,'Age':Age,'Address':Address}
    print("\nContact Added Successfully\n")

#View Contact
def viewcontact():
  Name=input("\nEnter name : ").strip()
  if Name in Contact:
    print("\n-- View Contact --")
    print("Name : ",Name)
    print("Phone : ",Contact[Name]['Phone'])
    print("Age : ",Contact[Name]['Age'])
    print("Address : ",Contact[Name]['Address'],'\n')
  else:
    print("Contact not found Please go to add the new contact \n")

#Update Contact
def updatecontact():
  Name=input("\nEnter name : ").strip()
  if Name not in Contact:
    print("Contact not found Please go to add the new contact \n")
  else:
     print("\n-- Please Update Contact --")
     Phone=input("Enter phone number : ").strip()
     Age=input("Enter age : ").strip()
     Address=input("Enter address : ").strip()
     Contact[Name]={"Phone":Phone,'Age':Age,'Address':Address}
     print("Contact Update Successfully\n")

#Delete Contcat
def deletecontact():
  Name=input("\nEnter name : ").strip()
  if Name not in Contact:
    print("Contact not found or Already Deleted \n")
  else:
    del Contact[Name]
    print("Contact Deleted successfully\n")

#View All Contact
def allcontact():
  if not Contact:
    print("Contact Book is empty.\n")
  else:
    for Name, Details in Contact.items():
      print("\n-- View All Contact --")
      print("Name : ",Name)
      print("Phone : ",Details['Phone'])
      print("Age : ",Details['Age'])
      print("Address : ",Details['Address'],'\n')

#Save Contact
def savecon():
  if Contact:
     df = pd.DataFrame(Contact).T.reset_index()
     df.columns = ['Name', ' Phone', ' Age', ' Address']
     df.to_csv("contacts.csv", index=False)
     print("All contacts saved to 'contacts.csv'.\n")
  else:
    print("No contacts to save.\n")

#Main Menu
def choice():
  while True:
    print("-- Welcome To Contact Dictionary --\n")
    print(" 1 . New Contact ")
    print(" 2 . View Contact ")
    print(" 3 . Update Contact ")
    print(" 4 . Delete Contact ")
    print(" 5 . All Contact ")
    print(" 6 . Save Contact ")
    print(" 7 . Exit \n")
    ch=input("Enter Your Choice : ").strip()
    if ch=="1":
      newcontact()
    elif ch=='2':
      viewcontact()
    elif ch=='3':
      updatecontact()
    elif ch=='4':
      deletecontact()
    elif ch=='5':
      allcontact()
    elif ch=="6":
      savecon()
    elif ch=='7':
      print("Good Bye\n")
      break
    else:
      print("\nInvalid choice. Please enter a number between 1 and 7.\n")
choice()


from google.colab import files
files.download("contacts.csv")