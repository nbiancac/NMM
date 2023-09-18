' Modi
'#include "vba_globals_all.lib"
Sub Main ()
	Dim mode_numbers As Integer
	Dim mode As Integer
	Dim Np As Integer
	Dim Length As Double
	Dim Pipe_radius As Double
	Dim sLabel As String

	sLabel = Units.GetGeometryUnit
	Length = InputBox("Enter the length of the structure [" & sLabel & "]:")
	Pipe_radius = InputBox("Enter the radius of the pipe [" & sLabel & "]:")
	Np = InputBox("Enter the number of points along the axis [insert number here]:")
	Start_mode = InputBox("Start counting from the mode [insert number here]:")
    mode_numbers = Solver.AKSGetNumberOfModes

    For mode = 1 To mode_numbers
        SelectTreeItem ("2D/3D Results\Modes\Mode" + Str(modo) + "\e")
        With ASCIIExport
            .Reset
            .FileName ("..\..\E_Mode_left" + Str(modo + Start_mode - 1) + ".txt")
            .SetSubvolume(-Pipe_radius, Pipe_radius, -Pipe_radius, Pipe_radius, 0., 0.)
            .Mode("FixedWidth")
            .Step(0.1)
            .Execute
        End With
        With ASCIIExport
            .Reset
            .FileName ("..\..\E_Mode_right" + Str(modo + Start_mode - 1) + ".txt")
            .SetSubvolume(-Pipe_radius, Pipe_radius, -Pipe_radius, Pipe_radius, Length, Length)
            .Mode("FixedWidth")
            .Step(0.1)
            .Execute
        End With
        With ASCIIExport
            .Reset
            .FileName ("..\..\Mode_on_axis" + Str(modo + Start_mode - 1) + ".txt")
            .SetSubvolume(0., 0., 0., 0., 0., Length)
            .Mode("FixedWidth")
            .Step(Length/Np)
            .Execute
        End With
        SelectTreeItem ("2D/3D Results\Modes\Mode" + Str(modo) + "\h")
        With ASCIIExport
            .Reset
            .FileName ("..\..\H_Mode_left" + Str(modo + Start_mode - 1) + ".txt")
            .SetSubvolume(-5., 5., -5., 5., 0., 0.)
            .Mode("FixedWidth")
            .Step(0.1)
            .Execute
        End With
		With ASCIIExport
            .Reset
            .FileName ("..\..\H_Mode_right" + Str(modo + Start_mode - 1) + ".txt")
            .SetSubvolume(-5., 5., -5., 5., Length, Length)
            .Mode("FixedWidth")
            .Step(0.1)
            .Execute
        End With
    Next

End Sub
