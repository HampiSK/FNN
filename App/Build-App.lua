project "App"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++latest"
   targetdir "Binaries/%{cfg.buildcfg}"
   staticruntime "off"

   files { "Source/**.h", "Source/**.c", "Source/**.hpp", "Source/**.cpp" }

   includedirs
   {
      -- Include FNN
      "../FNN/Source/Activation",
      "../FNN/Source/Edge",
      "../FNN/Source/Neuron",
      "../FNN/Source/NNetwork",
      "../FNN/Source/Random",

      -- Include Examples
      "Source/Example",

      -- Include Self
      "Source"
   }

   links
   {
      "FNN"
   }

   targetdir ("../Binaries/" .. OutputDir .. "/%{prj.name}")
   objdir ("../Binaries/Intermediates/" .. OutputDir .. "/%{prj.name}")

   filter "system:windows"
       systemversion "latest"
       defines { "WINDOWS" }

   filter "configurations:Debug"
       defines { "DEBUG" }
       runtime "Debug"
       symbols "On"

   filter "configurations:Release"
       defines { "RELEASE" }
       runtime "Release"
       optimize "On"
       symbols "On"

   filter "configurations:Dist"
       defines { "DIST" }
       runtime "Release"
       optimize "On"
       symbols "Off"