% Database containing names and dates of birth

person('SHREE', '12-05-2003').
person('Priya', '23-08-2004').
person('KADHOOL', '15-01-2002').
person('Divya', '09-11-2003').
person('Rahul', '30-06-2004').

% Rule to display a person's date of birth
show_dob(Name, DOB) :-
person(Name, DOB).
