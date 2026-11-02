% Facts
planet(mercury, 4879, rocky).
planet(venus, 12104, rocky).
planet(earth, 12742, rocky).
planet(mars, 6779, rocky).
planet(jupiter, 139820, gas_giant).
planet(saturn, 116460, gas_giant).
planet(uranus, 50724, ice_giant).
planet(neptune, 49244, ice_giant).

% Rules
terrestrial_planet(X) :-
    planet(X, _, rocky).

gas_planet(X) :-
    planet(X, _, gas_giant).

ice_planet(X) :-
    planet(X, _, ice_giant).

large_planet(X) :-
    planet(X, Diameter, _),
    Diameter > 50000.