library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package types is
	type img	is array(0 to 783) of std_logic;
	--signal test : img;
	type weightType	is array(0 to 1569) of std_logic_VECTOR(14 downto 0);
	--signal weight : weightType;
	type layerarray	is array(1 downto 0) of std_logic_vector(17 downto 0);

end package;

LIBRARY ieee  ; 
LIBRARY std  ; 
USE ieee.NUMERIC_STD.all  ; 
USE ieee.std_logic_1164.all  ; 
USE ieee.std_logic_textio.all  ; 
USE ieee.std_logic_unsigned.all  ; 
use work.types.all;

USE std.textio.all  ; 
ENTITY ann_tb  IS 
END ; 
 
ARCHITECTURE ann_tb_arch OF ann_tb IS
  component ann
port 
	(
		clk : in std_logic;
		output : out std_LOGIC;
		reset : in std_LOGIC;
		oSum : out std_LOGIC_VECTOR(17 downto 0);
		stout : out std_LOGIC_vector(2 downto 0);
		selecttest : in std_logic_vecTOR(1 downto 0);
		testO : out img;
		layerO: out layerarray
		
	);
	end component;
  SIGNAL output   :  STD_LOGIC  ; 
  SIGNAL oSum   : std_LOGIC_VECTOR(17 downto 0)  ; 
  signal stout  : std_LOGIC_vector(2 downto 0);
  SIGNAL clk    :  STD_LOGIC  ; 
  SIGNAL reset   :  STD_LOGIC:= '1'  ; 
  SIGNAL selecttest   :   std_logic_vecTOR(1 downto 0) ; 
  SIGNAL testO : img;
SIGNAL	layerO: layerarray;
  constant CLK_period : time := 10 ns;
  
  
BEGIN
  DUT  : ann  
    PORT MAP ( 
      output   => output  ,
		osum=>osum,
      clk   => clk  ,
      reset   => reset,
		stout	=> stout,
		selecttest =>selecttest,
		testO =>testO,
		layerO =>layerO
		) ; 

   CLK_process :process
   begin
		CLK <= '0';
		wait for CLK_period/2;
		CLK <= '1';
		wait for CLK_period/2;
   end process;
	
-- "Constant Pattern"
-- Start Time = 0 ns, End Time = 1 us, Period = 0 ns
  Process
	Begin
	selecttest<="00";
	wait for 10 ns ;
	reset <='0';

	wait for 6*CLK_period ;
	-------------------------
	selecttest<="01";
	reset<='1';
	wait for 10 ns ;
	reset <='0';

	wait for 6*CLK_period ;
	-------------------------
	selecttest<="10";
	reset<='1';
	wait for 10 ns ;
	reset <='0';
	wait for 6*CLK_period ;
	-------------------------
			assert false
          report "Simulation DONEEE!!!"
         severity failure;
 End Process;
END;
