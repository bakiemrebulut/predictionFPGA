library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity edgeDetection is

		port 
	(
		pixelIn	   				: in std_logic_vector  (0 downto 0);
		pixelOut 					: out std_logic_vector  (0 downto 0);
		
		edgeDetEn 	   			: in std_logic;
		address	  					: out std_logic_vector  (16 downto 0);
		edgeDetWE					: out std_logic:='0';
		EdgeDetRD					: out std_logic:='0';
		clk							: in std_logic;
		done							: out std_logic
	);

end entity;

architecture rtl of edgeDetection is
signal outPixel		:	std_logic_vector(0 downto 0);
signal wren		:	std_logic:='0';

type buff	is array(0 to 1,0 to 1) of std_logic_vector(0 downto 0);
signal frame : buff;

signal addressTemp : std_logic_vector(16 downto 0):=(others=>'0');
signal ifBreak : std_logic:='0';	


signal state : std_logic_vector(1 downto 0):="00";	

begin
done<=ifBreak;
address<=addressTemp;
pixelOut<=outPixel;
edgeDetWE<=wren;

filterProcess : process(clk,edgeDetEn)

variable counter : integer range 0 to 4:=0;
variable y : integer range 0 to 238:=0;
variable x : integer range 0 to 319:=0;

begin
if rising_edge(clk) then
		if edgeDetEn='1' then
			addressTemp<=(others=>'0');
			ifBreak<='0';
			state<="00";
			y:=0;
			x:=0;
			EdgeDetRD<='0';
			wren<='0';
		elsif ifBreak='0' then
			case state is
				when "00"=>--set and read 0,0
					if y=238 and x=318 then --set x,y & finish detection
						ifBreak<='1';
						wren<='0';
						EdgeDetRD<='0';
						addressTemp<=(others=>'0');
					elsif x=319 then
						y:=y+1;
						x:=0;
						state<="01";							
					else
						x:=x+1;
						state<="01";
					end if;
					EdgeDetRD<='1';
					wren<='0';
					addressTemp<=std_logic_vector(to_unsigned(x+y*320,addressTemp'length));
						-- y x
					frame(0,0)<=pixelIn;
					frame(0,1)<=frame(0,1);
					frame(1,0)<=frame(1,0);
					frame(1,1)<=frame(1,1);
				when "01"=>	--read 1,0
					addressTemp<=std_logic_vector(to_unsigned(x+1+y*320,addressTemp'length));
					frame(0,0)<=frame(0,0);
					frame(0,1)<=pixelIn;
					frame(1,0)<=frame(1,0);
					frame(1,1)<=frame(1,1);
					state<="10";
				when "10"=> --read 0,1
					addressTemp<=std_logic_vector(to_unsigned(x+(y+1)*320,addressTemp'length));
					frame(0,0)<=frame(0,0);
					frame(0,1)<=frame(0,1);
					frame(1,0)<=pixelIn;
					frame(1,1)<=frame(1,1);
					state<="11";
				when "11"=> --edge detect --gradient
					frame(0,0)<=frame(0,0);
					frame(0,1)<=frame(0,1);
					frame(1,0)<=frame(1,0);																				--   pixels
																																--    <x>
																																--   .....
					if (frame(1,0)>=frame(0,1) and frame(0,1)>=frame(0,0  ) ) then		--c>b>a        --  ^|c| |
						frame(1,1)<=std_logic_vector(unsigned(frame(1,0))-unsigned(frame(0,0)));      --  y|:::|
																																--  v|a|b|
					elsif (frame(0,1)>=frame(1,0) and frame(1,0)>=frame(0,0  ) )	then	--b>=c>=a   --	  '''''	                      
						frame(1,1)<=std_logic_vector(unsigned(frame(0,1))-unsigned(frame(0,0)));     

					elsif (frame(1,0)>=frame(0,0  ) and frame(0,0  )>=frame(0,1) )	then	--c>=a>=b
						if (unsigned(frame(0,0))-unsigned(frame(0,1))) > (unsigned(frame(1,0))-unsigned(frame(0,0))) then
							frame(1,1)<=std_logic_vector(unsigned(frame(0,0))-unsigned(frame(0,1)));
						else
							frame(1,1)<=std_logic_vector(unsigned(frame(1,0))-unsigned(frame(0,0)));     
						end if;
						
					elsif (frame(0,0  )>=frame(1,0) and frame(1,0)>=frame(0,1) )	then			--a>=c>=b
						frame(1,1)<=std_logic_vector(unsigned(frame(0,0))-unsigned(frame(0,1)));     

					elsif (frame(0,1)>=frame(0,0  ) and frame(0,0  )>=frame(1,0) )	then		--b>=a>=c
						if (unsigned(frame(0,0))-unsigned(frame(1,0))) > (unsigned(frame(0,1))-unsigned(frame(0,0))) then
							frame(1,1)<=std_logic_vector(unsigned(frame(0,0))-unsigned(frame(1,0)));
						else
							frame(1,1)<=std_logic_vector(unsigned(frame(0,1))-unsigned(frame(0,0)));     
						end if;
						
					elsif (frame(0,0  )>=frame(0,1) and frame(0,1)>=frame(1,0) )	then			--a>=b>=c
						frame(1,1)<=std_logic_vector(unsigned(frame(0,0))-unsigned(frame(1,0)));     

					end if;
					addressTemp<=std_logic_vector(to_unsigned(x+y*320,addressTemp'length));
					outPixel<=frame(1,1);--frame(1,1)(2)&frame(1,1)(2)&frame(1,1)(2);--(frame(1,1)(2)or(frame(1,1)(1)and frame(1,1)(0)))&(frame(1,1)(2)or(frame(1,1)(1)and frame(1,1)(0)))&(frame(1,1)(2)or(frame(1,1)(1)and frame(1,1)(0)));
					wren<='1';
					state<="00";
			end case;
		else
			wren<='0';
			EdgeDetRD<='0';
		end if;
end if;
end process;
end rtl;
